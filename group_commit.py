"""组提交 (group commit) 耐久性验证模拟器。

模拟数据库领导者将多个事务合成一批、一次 fsync 落盘、再复制到
远地副本的过程, 计算每个事务对客户端可见的"安全提交时间"。

核心区分:
  - 本地持久化 (local fsync): 领导者本机落盘, 此时若领导者宕机,
    未复制到多数派的数据可能丢失, 不能算耐久。
  - 多数派确认 (majority ack): 批次在多数派节点上持久化, 故障切换
    后仍可恢复, 才是对客户端可见的安全提交点。

领导者切换时, 未完成(未达到多数派)的批次必须重新判定:
  - 新领导者持有该批次 -> 重新 fsync + 复制, 重新计算可见时间。
  - 新领导者不持有该批次 -> 批次丢失, 事务对客户端不可见。
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

RISK_SAFE = "SAFE"
RISK_RECOVERED = "RECOVERED_AFTER_FAILOVER"
RISK_REBASED = "REBASED_ON_NEW_LEADER"
RISK_LOST = "LOST_ON_FAILOVER"


@dataclass
class Transaction:
    tx_id: str
    arrive: float
    deadline: float  # 客户端要求可见提交完成的绝对时间


@dataclass
class Replica:
    name: str
    delay: float  # 领导者 fsync 完成后, 复制到该副本并被确认的延迟


@dataclass
class Failover:
    time: float
    new_leader: str  # 被提升为新领导者的副本名


@dataclass
class Config:
    fsync_delay: float       # 一次批次 fsync 的耗时
    batch_max_size: int      # 批次最大事务数 (到达即关闭批次)
    batch_max_wait: float    # 批次最长等待时间 (防止小事务被无限推迟)
    replicas: List[Replica]

    @property
    def cluster_size(self):
        return 1 + len(self.replicas)

    @property
    def majority(self):
        return self.cluster_size // 2 + 1


@dataclass
class Batch:
    batch_id: int
    txns: List[Transaction]
    open_time: float
    close_time: float
    fsync_time: float = 0.0                # 当前领导者本地持久化完成时间
    majority_time: Optional[float] = None  # 多数派确认时间 (None = 丢失)
    lost: bool = False
    risk: str = RISK_SAFE


@dataclass
class TxnResult:
    tx_id: str
    arrive: float
    deadline: float
    batch_id: int
    leader_ack_time: Optional[float]   # 领导者本地落盘即返回的时间 (不安全)
    visible_time: Optional[float]      # 多数派确认的安全可见时间 (None = 丢失)
    batch_wait: float                  # 事务在批次内等待关闭的时间
    blocking_batch: int                # 阻断该事务的批次 id
    durability_risk: str
    deadline_missed: bool


def form_batches(config, transactions):
    # type: (Config, Sequence[Transaction]) -> List[Batch]
    """按到达顺序组批: 满 batch_max_size 立即关闭, 或等待
    batch_max_wait 后强制关闭, 保证小事务不被无限推迟。"""
    batches = []  # type: List[Batch]
    open_txns = []  # type: List[Transaction]
    open_time = [None]  # type: List[Optional[float]]

    def close(close_t):
        batches.append(Batch(
            batch_id=len(batches),
            txns=list(open_txns),
            open_time=open_time[0],
            close_time=close_t,
        ))

    for txn in sorted(transactions, key=lambda t: (t.arrive, t.tx_id)):
        if open_txns and open_time[0] + config.batch_max_wait <= txn.arrive:
            close(open_time[0] + config.batch_max_wait)
            open_txns = []
        if not open_txns:
            open_time[0] = txn.arrive
        open_txns.append(txn)
        if len(open_txns) >= config.batch_max_size:
            close(txn.arrive)
            open_txns = []
    if open_txns:
        close(open_time[0] + config.batch_max_wait)
    return batches


def _majority_time(config, fsync_time, replica_delays):
    # type: (Config, float, Sequence[float]) -> Optional[float]
    """领导者本地持久化算一票, 加上各副本确认票, 取第 majority 小的时间。"""
    votes = [fsync_time] + [fsync_time + d for d in replica_delays]
    if len(votes) < config.majority:
        return None
    votes.sort()
    return votes[config.majority - 1]


def _apply_failover(config, batches, failover):
    # type: (Config, List[Batch], Failover) -> None
    """领导者切换: 未达多数派的批次逐一重新判定, 不默认成功。"""
    remaining = [r for r in config.replicas if r.name != failover.new_leader]
    for batch in batches:
        if batch.lost or batch.majority_time is None:
            continue
        if batch.majority_time <= failover.time:
            continue  # 已达多数派, 切换不影响耐久性

        if batch.fsync_time > failover.time:
            # 旧领导者尚未落盘 (批次仍开放或 fsync 未完成):
            # 在新领导者上重排, fsync 从切换后重新计时。
            batch.fsync_time = max(batch.close_time, failover.time) + config.fsync_delay
            batch.majority_time = _majority_time(
                config, batch.fsync_time, [r.delay for r in remaining])
            batch.risk = RISK_REBASED
            if batch.majority_time is None:
                batch.lost = True
                batch.risk = RISK_LOST
            continue

        # 旧领导者已落盘但未达多数派: 检查新领导者是否持有该批次。
        holders = set()
        for r in config.replicas:
            if batch.fsync_time + r.delay <= failover.time:
                holders.add(r.name)
        if failover.new_leader not in holders:
            batch.lost = True
            batch.majority_time = None
            batch.risk = RISK_LOST
            continue

        # 新领导者持有批次: 重新 fsync 并向其余副本复制。
        new_fsync = failover.time + config.fsync_delay
        votes = [new_fsync]
        for r in remaining:
            if r.name in holders:
                votes.append(max(batch.fsync_time + r.delay, failover.time))
            else:
                votes.append(new_fsync + r.delay)
        votes.sort()
        batch.fsync_time = new_fsync
        batch.majority_time = votes[config.majority - 1] \
            if len(votes) >= config.majority else None
        batch.risk = RISK_RECOVERED
        if batch.majority_time is None:
            batch.lost = True
            batch.risk = RISK_LOST


def simulate(config, transactions, failovers=()):
    # type: (Config, Sequence[Transaction], Sequence[Failover]) -> List[TxnResult]
    """运行完整模拟, 返回每个事务的可见时间、阻断批次与耐久性风险。"""
    batches = form_batches(config, transactions)
    delays = [r.delay for r in config.replicas]
    for batch in batches:
        batch.fsync_time = batch.close_time + config.fsync_delay
        batch.majority_time = _majority_time(config, batch.fsync_time, delays)

    original_fsync = {b.batch_id: b.fsync_time for b in batches}
    for failover in sorted(failovers, key=lambda f: f.time):
        _apply_failover(config, batches, failover)

    results = []  # type: List[TxnResult]
    for batch in batches:
        for txn in batch.txns:
            visible = batch.majority_time
            results.append(TxnResult(
                tx_id=txn.tx_id,
                arrive=txn.arrive,
                deadline=txn.deadline,
                batch_id=batch.batch_id,
                leader_ack_time=original_fsync[batch.batch_id],
                visible_time=visible,
                batch_wait=batch.close_time - txn.arrive,
                blocking_batch=batch.batch_id,
                durability_risk=batch.risk,
                deadline_missed=visible is None or visible > txn.deadline,
            ))
    results.sort(key=lambda r: (r.arrive, r.tx_id))
    return results


def format_report(results):
    # type: (Sequence[TxnResult]) -> str
    header = ("tx_id", "arrive", "batch", "leader_ack", "visible",
              "wait", "deadline", "risk", "deadline_missed")
    rows = [header]
    for r in results:
        rows.append((
            r.tx_id,
            "%.2f" % r.arrive,
            str(r.blocking_batch),
            "%.2f" % r.leader_ack_time if r.leader_ack_time is not None else "-",
            "%.2f" % r.visible_time if r.visible_time is not None else "LOST",
            "%.2f" % r.batch_wait,
            "%.2f" % r.deadline,
            r.durability_risk,
            "YES" if r.deadline_missed else "no",
        ))
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    return "\n".join(
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        for row in rows
    )


def _demo():
    config = Config(
        fsync_delay=1.0,
        batch_max_size=3,
        batch_max_wait=4.0,
        replicas=[Replica("r1", 0.5), Replica("r2", 6.0)],
    )
    txns = [
        Transaction("tx-a", 0.0, 20.0),
        Transaction("tx-b", 0.5, 20.0),
        Transaction("tx-c", 0.8, 20.0),
        Transaction("tx-d", 1.0, 6.0),
        Transaction("tx-e", 9.0, 30.0),
    ]
    failovers = [Failover(time=2.0, new_leader="r1")]
    print(format_report(simulate(config, txns, failovers)))


if __name__ == "__main__":
    _demo()
