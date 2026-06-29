source /home/xufei/miniconda3/etc/profile.d/conda.sh
conda activate acent

python3 << 'EOF'
import lmdb
p = "/home/xufei/tibet_acent/parseq_lmdb/train"
env = lmdb.open(p, readonly=True, lock=False)
txn = env.begin()
n = txn.get(b"num-samples")
lb1 = txn.get(b"label-000001")
lb2 = txn.get(b"label-000002")
lb3 = txn.get(b"label-000003")
print("num_samples:", n)
print("label-000001:", lb1)
print("label-000002:", lb2)
print("label-000003:", lb3)
env.close()

# check val
p2 = "/home/xufei/tibet_acent/parseq_lmdb/val"
env2 = lmdb.open(p2, readonly=True, lock=False)
txn2 = env2.begin()
n2 = txn2.get(b"num-samples")
lb_v1 = txn2.get(b"label-000001")
print("val num_samples:", n2)
print("val label-000001:", lb_v1)
env2.close()
EOF
