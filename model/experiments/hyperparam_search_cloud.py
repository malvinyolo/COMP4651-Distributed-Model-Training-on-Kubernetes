#!/usr/bin/env python3
"""
Parallel Hyperparameter Search on Kubernetes

This script orchestrates parallel hyperparameter search by:
1. Generating multiple config files for different hyperparameter combinations
2. Creating separate Kubernetes jobs for each configuration
3. Running all jobs in parallel across the cluster
4. Collecting and comparing results

This demonstrates cloud capability: parallel execution of multiple experiments
that would take 6+ minutes sequentially on a single machine.
"""

import os
import yaml
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Hyperparameter search space
SEARCH_SPACE = {
    "learning_rate": [1e-4, 5e-4, 1e-3],
    "hidden_size": [64, 128, 256],
    "num_layers": [2, 3]
}

def generate_config(base_config: Dict, params: Dict, output_dir: Path) -> Path:
    """Generate a config file for specific hyperparameter combination."""
    config = base_config.copy()
    
    # Update with search parameters
    config["model"]["hidden_size"] = params["hidden_size"]
    config["model"]["num_layers"] = params["num_layers"]
    config["training"]["learning_rate"] = params["learning_rate"]
    
    # Create unique run name
    run_name = f"hp_search_lr{params['learning_rate']}_h{params['hidden_size']}_l{params['num_layers']}"
    config["run_name"] = run_name
    
    # Save config
    config_path = output_dir / f"{run_name}.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path

def create_kubernetes_job(config_path: Path, job_name: str, namespace: str = "stock-training") -> str:
    """Create a Kubernetes PyTorchJob manifest for a single hyperparameter config."""
    job_manifest = f"""apiVersion: "kubeflow.org/v1"
kind: PyTorchJob
metadata:
  name: {job_name}
  namespace: {namespace}
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: stock-trainer:latest
              imagePullPolicy: IfNotPresent
              command:
                - python
                - -m
                - src.cli
                - train
                - --config
                - /data/{config_path.name}
              volumeMounts:
                - name: data
                  mountPath: /data
                - name: outputs
                  mountPath: /outputs
              resources:
                limits:
                  cpu: "2"
                  memory: "4Gi"
                requests:
                  cpu: "1"
                  memory: "2Gi"
          volumes:
            - name: data
              persistentVolumeClaim:
                claimName: stock-data-pvc
            - name: outputs
              emptyDir: {{}}
"""
    
    # Save manifest
    manifest_path = config_path.parent / f"{job_name}.yaml"
    with open(manifest_path, 'w') as f:
        f.write(job_manifest)
    
    return str(manifest_path)

def submit_job(manifest_path: str) -> bool:
    """Submit a Kubernetes job."""
    try:
        result = subprocess.run(
            ["kubectl", "apply", "-f", manifest_path],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✓ Submitted: {manifest_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to submit {manifest_path}: {e.stderr}")
        return False

def wait_for_jobs(job_names: List[str], namespace: str = "stock-training", timeout: int = 600) -> Dict[str, str]:
    """Wait for all jobs to complete and return their statuses."""
    print(f"\nWaiting for {len(job_names)} jobs to complete (timeout: {timeout}s)...")
    
    start_time = time.time()
    statuses = {}
    
    while time.time() - start_time < timeout:
        all_done = True
        
        for job_name in job_names:
            if job_name in statuses:
                continue
            
            try:
                result = subprocess.run(
                    ["kubectl", "get", "pytorchjob", job_name, "-n", namespace, "-o", "json"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                job_info = json.loads(result.stdout)
                conditions = job_info.get("status", {}).get("conditions", [])
                
                for condition in conditions:
                    if condition["type"] == "Succeeded" and condition["status"] == "True":
                        statuses[job_name] = "Succeeded"
                        print(f"✓ {job_name}: Succeeded")
                        break
                    elif condition["type"] == "Failed" and condition["status"] == "True":
                        statuses[job_name] = "Failed"
                        print(f"✗ {job_name}: Failed")
                        break
                else:
                    all_done = False
                    
            except subprocess.CalledProcessError:
                all_done = False
        
        if all_done:
            break
        
        time.sleep(10)
    
    # Mark remaining jobs as timeout
    for job_name in job_names:
        if job_name not in statuses:
            statuses[job_name] = "Timeout"
            print(f"⏱ {job_name}: Timeout")
    
    return statuses

def collect_results(job_names: List[str], namespace: str = "stock-training") -> List[Dict]:
    """Collect results from completed jobs."""
    results = []
    
    for job_name in job_names:
        try:
            # Get pod name for the job
            result = subprocess.run(
                ["kubectl", "get", "pods", "-n", namespace, "-l", f"pytorch-job-name={job_name}", 
                 "-o", "jsonpath={.items[0].metadata.name}"],
                capture_output=True,
                text=True,
                check=True
            )
            pod_name = result.stdout.strip()
            
            if not pod_name:
                continue
            
            # Get logs
            result = subprocess.run(
                ["kubectl", "logs", pod_name, "-n", namespace],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse metrics from logs (looking for final metrics output)
            logs = result.stdout
            # You would parse the actual metrics here based on your training output format
            # For now, we'll create a placeholder
            results.append({
                "job_name": job_name,
                "logs": logs,
                "metrics": {}  # Parse from logs
            })
            
        except subprocess.CalledProcessError as e:
            print(f"Failed to get results for {job_name}: {e.stderr}")
    
    return results

def main():
    """Main execution function."""
    print("=" * 80)
    print("Parallel Hyperparameter Search on Kubernetes")
    print("=" * 80)
    
    # Setup directories
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "experiments" / "hp_search_cloud"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load base config
    base_config_path = project_root / "config_test.yaml"
    with open(base_config_path) as f:
        base_config = yaml.safe_load(f)
    
    # Generate all hyperparameter combinations
    combinations = []
    for lr in SEARCH_SPACE["learning_rate"]:
        for hs in SEARCH_SPACE["hidden_size"]:
            for nl in SEARCH_SPACE["num_layers"]:
                combinations.append({
                    "learning_rate": lr,
                    "hidden_size": hs,
                    "num_layers": nl
                })
    
    print(f"\nGenerating configs for {len(combinations)} hyperparameter combinations...")
    
    # Generate configs and job manifests
    job_names = []
    manifest_paths = []
    
    for i, params in enumerate(combinations):
        config_path = generate_config(base_config, params, output_dir)
        job_name = f"hp-search-{i:02d}"
        manifest_path = create_kubernetes_job(config_path, job_name)
        
        job_names.append(job_name)
        manifest_paths.append(manifest_path)
        
        print(f"  {i+1:2d}. {config_path.name}")
    
    # Submit all jobs
    print(f"\nSubmitting {len(job_names)} jobs to Kubernetes...")
    start_time = time.time()
    
    for manifest_path in manifest_paths:
        submit_job(manifest_path)
    
    submit_time = time.time() - start_time
    print(f"\nAll jobs submitted in {submit_time:.2f}s")
    
    # Wait for completion
    statuses = wait_for_jobs(job_names)
    total_time = time.time() - start_time
    
    # Collect results
    print("\nCollecting results...")
    results = collect_results(job_names)
    
    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_combinations": len(combinations),
        "total_time_seconds": total_time,
        "submit_time_seconds": submit_time,
        "statuses": statuses,
        "results": results
    }
    
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total combinations: {len(combinations)}")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Average time per config: {total_time/len(combinations):.2f}s")
    print(f"\nStatus breakdown:")
    for status, count in sorted(Counter(statuses.values()).items()):
        print(f"  {status}: {count}")
    print(f"\nResults saved to: {summary_path}")
    print("=" * 80)

if __name__ == "__main__":
    from collections import Counter
    main()
