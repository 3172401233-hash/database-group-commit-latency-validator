# -*- coding: utf-8 -*-
"""group_commit 模拟器的单元测试 (Python 3.8.2 兼容)。"""

import unittest

from group_commit import (
    Config,
    Failover,
    Replica,
    RISK_LOST,
    RISK_REBASED,
    RISK_RECOVERED,
    RISK_SAFE,
    Transaction,
    form_batches,
    simulate,
)


def make_config(**overrides):
    params = dict(
        fsync_delay=1.0,
        batch_max_size=2,
        batch_max_wait=5.0,
        replicas=[Replica("r1", 1.0), Replica("r2", 1.0)],
    )
    params.update(overrides)
    return Config(**params)


def by_id(results):
    return {r.tx_id: r for r in results}


class BatchCrossingTest(unittest.TestCase):
    """批次交叉: 满批立即关闭, 后续事务进入下一批次。"""

    def test_full_batch_closes_and_next_batch_opens(self):
        config = make_config()
        txns = [
            Transaction("tx-1", 0.0, 100.0),
            Transaction("tx-2", 0.5, 100.0),
            Transaction("tx-3", 0.6, 100.0),
        ]
        results = by_id(simulate(config, txns))
        # 批次 0: tx-1, tx-2 (0.5 时满批关闭), fsync 1.5, 多数派 2.5
        self.assertEqual(results["tx-1"].blocking_batch, 0)
        self.assertEqual(results["tx-2"].blocking_batch, 0)
        self.assertAlmostEqual(results["tx-1"].visible_time, 2.5)
        # 批次 1: tx-3 独自等待 max_wait, 5.6 关闭, fsync 6.6, 多数派 7.6
        self.assertEqual(results["tx-3"].blocking_batch, 1)
        self.assertAlmostEqual(results["tx-3"].visible_time, 7.6)
        self.assertEqual(results["tx-3"].durability_risk, RISK_SAFE)

    def test_small_transaction_not_deferred_beyond_max_wait(self):
        config = make_config(batch_max_size=10, batch_max_wait=5.0)
        txns = [Transaction("tx-lonely", 10.0, 100.0)]
        (result,) = simulate(config, txns)
        # 小事务最多等待 max_wait, 不能为了凑批被无限推迟
        self.assertLessEqual(result.batch_wait, 5.0)
        # close=15, fsync=16, 多数派=17
        self.assertAlmostEqual(result.visible_time, 17.0)
        self.assertAlmostEqual(result.leader_ack_time, 16.0)

    def test_batch_boundary_at_exact_max_wait(self):
        config = make_config(batch_max_size=10, batch_max_wait=5.0)
        txns = [
            Transaction("tx-a", 0.0, 100.0),
            Transaction("tx-b", 5.0, 100.0),
        ]
        batches = form_batches(config, txns)
        self.assertEqual(len(batches), 2)
        self.assertEqual([t.tx_id for t in batches[0].txns], ["tx-a"])
        self.assertEqual([t.tx_id for t in batches[1].txns], ["tx-b"])
class SlowReplicaTest(unittest.TestCase):
    """慢副本: 多数派由最快的足够副本决定, 慢副本不阻塞可见性。"""

    def test_slow_replica_does_not_block_majority(self):
        config = make_config(
            batch_max_size=1,
            replicas=[Replica("r1", 0.5), Replica("r2", 0.5),
                      Replica("r3", 0.5), Replica("r4-slow", 9.0)],
        )
        self.assertEqual(config.majority, 3)
        (result,) = simulate(config, [Transaction("tx-1", 0.0, 100.0)])
        # fsync=1.0, 票: 1.0(leader), 1.5, 1.5, 1.5, 10.0 -> 第 3 小为 1.5
        self.assertAlmostEqual(result.visible_time, 1.5)

    def test_majority_forced_to_wait_for_slow_replica(self):
        config = make_config(
            batch_max_size=1,
            replicas=[Replica("r1", 0.5), Replica("r2-slow", 9.0),
                      Replica("r3-slow", 9.0)],
        )
        self.assertEqual(config.majority, 3)
        (result,) = simulate(config, [Transaction("tx-1", 0.0, 100.0)])
        # 票: 1.0(leader), 1.5, 10.0, 10.0 -> 第 3 小为 10.0
        self.assertAlmostEqual(result.visible_time, 10.0)
        # 本地落盘 1.0 与可见 10.0 的差距即耐久性风险窗口
        self.assertAlmostEqual(
            result.visible_time - result.leader_ack_time, 9.0)


class LeaderFailoverTest(unittest.TestCase):
    """领导者切换: 未完成批次重新判定, 不默认成功。"""

    def test_batch_only_on_dead_leader_is_lost(self):
        config = make_config(
            batch_max_size=1,
            replicas=[Replica("r1", 5.0), Replica("r2", 5.0)],
        )
        # fsync=1.0, 副本 6.0 才收到; t=2.0 切换时只有旧领导者持有
        (result,) = simulate(
            config, [Transaction("tx-1", 0.0, 100.0)],
            [Failover(time=2.0, new_leader="r1")],
        )
        self.assertIsNone(result.visible_time)
        self.assertEqual(result.durability_risk, RISK_LOST)
        self.assertTrue(result.deadline_missed)

    def test_batch_on_new_leader_is_recovered(self):
        config = make_config(
            batch_max_size=1,
            replicas=[Replica("r1", 0.5), Replica("r2", 5.0),
                      Replica("r3", 5.0), Replica("r4", 5.0)],
        )
        # 多数派为 3; fsync=1.0, r1 于 1.5 持有, 多数派需等到 6.0;
        # t=2.0 切换时批次未达多数派, r1 升任新领导者
        (result,) = simulate(
            config, [Transaction("tx-1", 0.0, 100.0)],
            [Failover(time=2.0, new_leader="r1")],
        )
        # 新领导者重新 fsync: 3.0; 其余副本 8.0 确认 -> 多数派 8.0
        self.assertEqual(result.durability_risk, RISK_RECOVERED)
        self.assertAlmostEqual(result.visible_time, 8.0)

    def test_durable_batch_unaffected_by_failover(self):
        config = make_config(batch_max_size=1)
        # fsync=1.0, 多数派=2.0; t=5.0 切换时批次已耐久
        (result,) = simulate(
            config, [Transaction("tx-1", 0.0, 100.0)],
            [Failover(time=5.0, new_leader="r1")],
        )
        self.assertEqual(result.durability_risk, RISK_SAFE)
        self.assertAlmostEqual(result.visible_time, 2.0)

    def test_open_batch_rebased_on_new_leader(self):
        config = make_config(batch_max_size=10, batch_max_wait=10.0)
        # 批次 0.0 开启, 10.0 才关闭; t=3.0 切换时旧领导者尚未落盘
        (result,) = simulate(
            config, [Transaction("tx-1", 0.0, 100.0)],
            [Failover(time=3.0, new_leader="r1")],
        )
        # 重新判定而非默认成功: close=10, fsync=11, 多数派=12
        self.assertEqual(result.durability_risk, RISK_REBASED)
        self.assertAlmostEqual(result.visible_time, 12.0)
        self.assertGreater(result.visible_time, 3.0)


class DeadlineTest(unittest.TestCase):
    """deadline 过期: 可见时间超过 deadline 即判定错过。"""

    def test_deadline_expired_when_replication_slow(self):
        config = make_config(
            batch_max_size=1,
            replicas=[Replica("r1", 0.5), Replica("r2-slow", 9.0),
                      Replica("r3-slow", 9.0)],
        )
        txns = [
            Transaction("tx-tight", 0.0, 5.0),
            Transaction("tx-loose", 0.1, 100.0),
        ]
        results = by_id(simulate(config, txns))
        # 两事务同批, 可见时间 10.0
        self.assertTrue(results["tx-tight"].deadline_missed)
        self.assertFalse(results["tx-loose"].deadline_missed)
        self.assertAlmostEqual(results["tx-tight"].visible_time, 10.0)

    def test_lost_transaction_counts_as_deadline_missed(self):
        config = make_config(
            batch_max_size=1,
            replicas=[Replica("r1", 5.0), Replica("r2", 5.0)],
        )
        (result,) = simulate(
            config, [Transaction("tx-1", 0.0, 1000.0)],
            [Failover(time=2.0, new_leader="r1")],
        )
        self.assertIsNone(result.visible_time)
        self.assertTrue(result.deadline_missed)


if __name__ == "__main__":
    unittest.main()
