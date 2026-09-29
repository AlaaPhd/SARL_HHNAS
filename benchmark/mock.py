import hashlib
class MockBenchmark:
    """Deterministic software-test benchmark. Never use its values as paper results."""
    def _all(self,arch,dataset):
        key=(arch.to_nats_string()+dataset).encode()
        z=int(hashlib.sha256(key).hexdigest()[:12],16)
        val=70.0+(z%2500)/100.0
        test=val-0.2+((z//2500)%40)/100.0
        params=0.2+((z//100000)%500)/100.0
        flops=10.0+((z//10000000)%15000)/100.0
        return min(val,94.99),min(test,94.99),params,flops
    def evaluate(self,arch,dataset="cifar10"):
        val,_,params,flops=self._all(arch,dataset)
        return {"val_acc":val,"params":params,"flops":flops,"cost":1.0}
    def final_test(self,arch,dataset="cifar10"):
        _,test,_,_=self._all(arch,dataset); return test
