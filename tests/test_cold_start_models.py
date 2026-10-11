import unittest
import numpy as np
from src.backbone.improve_s3 import pair_features, tanimoto, transfer


class ColdStartModelsTests(unittest.TestCase):
    def test_pair_features_are_swap_invariant(self):
        z=np.random.default_rng(10).normal(size=(8,5))
        pairs=np.asarray([[1,3],[4,2]])
        sim=tanimoto((z>0).astype(float))
        np.testing.assert_array_equal(pair_features(z,pairs,[sim]),pair_features(z,pairs[:,::-1],[sim]))

    def test_transfer_uses_training_reference_only(self):
        sim=np.eye(5)+.2
        adj=np.asarray([[0,1,0],[1,0,1],[0,1,0.]])
        score=transfer(sim,np.asarray([0,1,2]),adj,2)
        self.assertEqual(score.shape,(5,5))
        np.testing.assert_allclose(score,score.T,atol=1e-12)
        self.assertTrue(np.all((score>=0)&(score<=1)))
        # Alter query-query similarity: no held-out edges enter the training graph.
        changed=sim.copy();changed[3:,3:]=100
        np.testing.assert_allclose(score,transfer(changed,np.asarray([0,1,2]),adj,2))


if __name__=='__main__':unittest.main()
