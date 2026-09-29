
class NATSBenchTSS:
    """Leakage-safe NATS-Bench TSS adapter.

    Search calls expose validation accuracy only. Test accuracy is available
    exclusively through final_test(), intended to be called after search ends.
    """
    DATASET_MAP = {
        "cifar10": ("cifar10-valid", "cifar10"),
        "cifar10-valid": ("cifar10-valid", "cifar10"),
        "cifar100": ("cifar100", "cifar100"),
        "ImageNet16-120": ("ImageNet16-120", "ImageNet16-120"),
        "imagenet16-120": ("ImageNet16-120", "ImageNet16-120"),
    }

    def __init__(self, path, hp="200", fast_mode=True, verbose=False, trial_seed=False):
        try:
            from nats_bench import create
        except ImportError as e:
            raise ImportError("Install with: pip install nats_bench") from e
        self.api = create(path, "tss", fast_mode=fast_mode, verbose=verbose)
        self.hp = str(hp)
        self.trial_seed = trial_seed
        self.search_queries = 0
        self.test_queries = 0

    def _index(self, arch):
        s = arch.to_nats_string() if hasattr(arch, "to_nats_string") else arch
        idx = self.api.query_index_by_arch(s)
        if idx is None or idx < 0:
            raise ValueError(f"Architecture not found in TSS: {s}")
        return idx

    def _datasets(self, dataset):
        if dataset not in self.DATASET_MAP:
            raise ValueError(f"Unsupported dataset: {dataset}")
        return self.DATASET_MAP[dataset]

    def evaluate(self, arch, dataset="cifar10"):
        """SEARCH-TIME query. Never returns a test metric."""
        idx = self._index(arch)
        search_ds, cost_ds = self._datasets(dataset)
        info = self.api.get_more_info(
            idx, search_ds, hp=self.hp, is_random=self.trial_seed
        )
        if "valid-accuracy" not in info:
            raise RuntimeError(
                f"NATS-Bench returned no validation accuracy for {search_ds}. "
                "Search is stopped rather than falling back to test accuracy."
            )
        cost = self.api.get_cost_info(idx, cost_ds, hp=self.hp)
        self.search_queries += 1
        valid_time = info.get("valid-all-time", info.get("valid-per-time", 0.0))
        return {
            "val_acc": float(info["valid-accuracy"]),
            "params": float(cost.get("params", 0.0)),
            "flops": float(cost.get("flops", 0.0)),
            "latency": float(cost.get("latency", 0.0) or 0.0),
            "cost": float(info.get("train-all-time", 0.0) + valid_time),
            "arch_index": int(idx),
        }

    def final_test(self, arch, dataset="cifar10"):
        """POST-SEARCH query. Call only for the architecture selected by validation."""
        idx = self._index(arch)
        search_ds, _ = self._datasets(dataset)
        meta = self.api.query_meta_info_by_index(idx, hp=self.hp)
        if dataset in ("cifar10", "cifar10-valid"):
            metric = meta.get_metrics("cifar10", "ori-test", None, False)
        elif dataset == "cifar100":
            metric = meta.get_metrics("cifar100", "x-test", None, False)
        else:
            metric = meta.get_metrics("ImageNet16-120", "x-test", None, False)
        self.test_queries += 1
        return float(metric["accuracy"])

    def reference_best_validation(self, dataset="cifar10"):
        """Exhaustive benchmark reference; use for post-hoc regret only, never search."""
        best = float("-inf")
        # TSS has 15,625 candidates.
        for idx in range(15625):
            search_ds, _ = self._datasets(dataset)
            info = self.api.get_more_info(idx, search_ds, hp=self.hp, is_random=False)
            if "valid-accuracy" in info:
                best = max(best, float(info["valid-accuracy"]))
        return best
