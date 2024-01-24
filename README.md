## Lora
```bash
python3 lora_linear.py --config-path scripts/linear/cifar --config-name lora.yaml

python3 lora_linear.py --config-path scripts/linear/cifar --config-name lora_cifar_100.yaml
```
## Autoattack Test
```bash
python3 aa_test.py --config-path scripts/aatest/cifar --config-name lora.yaml
```
```bash
python3 aa_test.py --config-path scripts/aatest/cifar --config-name lora_100.yaml
```

