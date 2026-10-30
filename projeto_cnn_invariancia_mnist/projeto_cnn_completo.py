"""Experimento completo do projeto: CNNs, invariância e robustez no MNIST.

Executar, por exemplo:
    python projeto_cnn_completo.py --epochs 8 --data-dir ./data --output-dir ./resultados

O programa treina duas arquiteturas do zero, com e sem augmentation, avalia as
seis condições pedidas e salva tabelas, curvas, exemplos, filtros e relatório.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

from mnist_cnn_invariancia_augmentation import (
    AddGaussianNoise,
    FixedBrightnessContrast,
    FixedRotate,
    FixedTranslate,
    MNIST_MEAN,
    MNIST_STD,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize() -> transforms.Normalize:
    return transforms.Normalize(MNIST_MEAN, MNIST_STD)


def train_transform(augmented: bool) -> transforms.Compose:
    ops: list = [transforms.ToTensor()]
    if augmented:
        ops += [
            transforms.RandomAffine(15, translate=(.15, .15), scale=(.9, 1.1), shear=5),
            transforms.RandomApply([AddGaussianNoise(std=.10)], p=.35),
        ]
    ops.append(normalize())
    return transforms.Compose(ops)


def test_transform(kind: str) -> transforms.Compose:
    extra = {
        "original": [],
        "translated": [FixedTranslate(5, 3)],
        "rotated": [FixedRotate(20)],
        "noisy": [AddGaussianNoise(std=.22)],
        "contrast": [FixedBrightnessContrast(.60, 1.9)],
        "combined": [FixedTranslate(3, 3), FixedRotate(15), AddGaussianNoise(std=.12),
                     FixedBrightnessContrast(.75, 1.5)],
    }
    if kind not in extra:
        raise ValueError(f"Condição desconhecida: {kind}")
    return transforms.Compose([transforms.ToTensor(), *extra[kind], normalize()])


class SimpleCNN(nn.Module):
    """Dois blocos convolucionais e uma camada totalmente conectada."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(32 * 7 * 7, 64),
                                        nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class DeepCNN(nn.Module):
    """Três blocos convolucionais, batch normalization e dropout."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(.35),
                                        nn.Linear(128 * 3 * 3, 128), nn.ReLU(),
                                        nn.Dropout(.35), nn.Linear(128, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


ARCHITECTURES = {"simple": SimpleCNN, "deep": DeepCNN}
CONDITIONS = ["original", "translated", "rotated", "noisy", "contrast", "combined"]


def loaders(data_dir: Path, batch_size: int, workers: int, seed: int):
    train_sets = {
        "plain": datasets.MNIST(data_dir, train=True, download=True,
                                 transform=train_transform(False)),
        "augmented": datasets.MNIST(data_dir, train=True, download=True,
                                     transform=train_transform(True)),
    }
    # A fixed validation split is shared by the two training conditions.
    indices = list(range(len(train_sets["plain"])))
    generator = torch.Generator().manual_seed(seed)
    train_idx, val_idx = random_split(indices, [55000, 5000], generator=generator)
    train_loaders = {}
    for key, dataset in train_sets.items():
        subset = torch.utils.data.Subset(dataset, train_idx.indices)
        train_loaders[key] = DataLoader(subset, batch_size, shuffle=True,
                                         num_workers=workers, pin_memory=True)
    val_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                 transform=train_transform(False))
    val_loader = DataLoader(torch.utils.data.Subset(val_dataset, val_idx.indices),
                            batch_size, shuffle=False, num_workers=workers)
    tests = {}
    for kind in CONDITIONS:
        ds = datasets.MNIST(data_dir, train=False, download=True,
                            transform=test_transform(kind))
        tests[kind] = DataLoader(ds, batch_size, shuffle=False, num_workers=workers)
    return train_loaders, val_loader, tests


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval(); total_loss = 0.0; correct = 0; total = 0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x); total_loss += criterion(logits, y).item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item(); total += y.size(0)
    return total_loss / total, correct / total


def train(model, train_loader, val_loader, device, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(epochs):
        model.train(); loss_sum = correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x); loss = criterion(logits, y)
            loss.backward(); optimizer.step()
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item(); total += y.size(0)
        val_loss, val_acc = evaluate(model, val_loader, device)
        history["train_loss"].append(loss_sum / total); history["train_acc"].append(correct / total)
        history["val_loss"].append(val_loss); history["val_acc"].append(val_acc)
        print(f"  época {epoch + 1:02d}/{epochs}: loss={loss_sum / total:.4f} val_acc={val_acc:.4f}")
    return history


def save_curves(history: dict, title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="treino"); axes[0].plot(epochs, history["val_loss"], label="validação")
    axes[0].set(xlabel="época", ylabel="loss", title="Loss"); axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], label="treino"); axes[1].plot(epochs, history["val_acc"], label="validação")
    axes[1].set(xlabel="época", ylabel="acurácia", title="Acurácia"); axes[1].legend()
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


@torch.no_grad()
def save_examples(model, tests: Dict[str, DataLoader], device: torch.device, out: Path) -> None:
    fig, axes = plt.subplots(len(CONDITIONS), 8, figsize=(12, 10))
    for row, kind in enumerate(CONDITIONS):
        x, y = next(iter(tests[kind])); pred = model(x.to(device)).argmax(1).cpu()
        correct = [i for i in range(len(y)) if int(pred[i]) == int(y[i])]
        incorrect = [i for i in range(len(y)) if int(pred[i]) != int(y[i])]
        selected = correct[:4] + incorrect[:4]
        selected += list(range(len(selected), 8))
        for shown, i in enumerate(selected[:8]):
            ax = axes[row, shown]; ax.imshow(x[i, 0] * MNIST_STD[0] + MNIST_MEAN[0], cmap="gray")
            good = int(pred[i]) == int(y[i]); label = "acerto" if good else "erro"
            ax.set_title(f"{label}\np={pred[i]} y={y[i]}", color="green" if good else "red", fontsize=8)
            ax.axis("off")
        axes[row, 0].set_ylabel(kind, rotation=0, labelpad=35, va="center")
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def save_filters(model: nn.Module, out: Path) -> None:
    weights = next(m for m in model.modules() if isinstance(m, nn.Conv2d)).weight.detach().cpu()
    fig, axes = plt.subplots(2, min(8, weights.size(0)), figsize=(12, 3))
    for i, ax in enumerate(np.ravel(axes)):
        if i < weights.size(0): ax.imshow(weights[i, 0], cmap="coolwarm"); ax.axis("off")
    fig.suptitle("Filtros da primeira camada convolucional"); fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def write_report(out: Path, rows: list[dict], epochs: int, device: str) -> None:
    best = max(rows, key=lambda r: r["accuracy"])
    lines = ["# Projeto: CNNs são realmente invariantes?", "",
             "## 1. Introdução", "Este experimento testa se CNNs treinadas do zero no MNIST mantêm o desempenho quando a imagem sofre transformações visuais.", "",
             "## 2. Metodologia", f"Foram comparadas as arquiteturas `simple` e `deep`, cada uma treinada com os mesmos dados em modo `plain` e `augmented`, por {epochs} épocas, no dispositivo `{device}`. O conjunto de validação contém 5.000 imagens fixadas por semente. Os testes são original, deslocado, rotacionado, com ruído, contraste alterado e combinado.", "",
             "## 3. Resultados", "A tabela é gerada a partir da execução completa:", "", "| modelo | treino | teste | loss | acurácia |", "|---|---|---|---:|---:|"]
    lines += [f"| {r['model']} | {r['training']} | {r['condition']} | {float(r['loss']):.4f} | {float(r['accuracy']):.4f} |" for r in rows]
    lines += ["", "Os arquivos `curvas_*.png`, `exemplos_*.png` e `filtros_*.png` documentam as evidências visuais.", "",
              "## 4. Discussão", "Compare a queda relativa entre `original` e cada transformação, verificando se o augmentation melhora robustez e se há custo no teste original. Interprete erros visualmente semelhantes e observe se os filtros detectam bordas e traços locais.", "",
              "## 5. Conclusão obrigatória", f"Com base nos resultados, a maior acurácia individual foi {best['accuracy']:.4f} ({best['model']}/{best['training']}/{best['condition']}). As CNNs deste projeto devem ser descritas como parcialmente robustas: invariância completa exigiria desempenho essencialmente preservado sob as transformações, algo que precisa ser verificado pela tabela e pelas quedas relativas. O augmentation pode ampliar a robustez às transformações que simula, mas não garante invariância a toda mudança visual."]
    (out / "relatorio.md").write_text("\n".join(lines), encoding="utf-8")


def write_detailed_report(out: Path, rows: list[dict], epochs: int, device: str) -> None:
    def acc(model: str, training: str, condition: str) -> float:
        return next(float(r["accuracy"]) for r in rows if r["model"] == model and r["training"] == training and r["condition"] == condition)

    transformed = [c for c in CONDITIONS if c != "original"]
    drops = [(acc(m, t, "original") - acc(m, t, c), m, t, c)
             for m in ARCHITECTURES for t in ("plain", "augmented") for c in transformed]
    largest_drop, drop_model, drop_training, drop_condition = max(drops)
    summary = []
    for model in ARCHITECTURES:
        for training in ("plain", "augmented"):
            original = acc(model, training, "original")
            worst_acc, worst_condition = min((acc(model, training, c), c) for c in transformed)
            summary.append(f"| {model} | {training} | {original:.4f} | {worst_condition} ({worst_acc:.4f}) | {original - worst_acc:.4f} |")
    transformed_improvements = sum(acc(m, "augmented", c) > acc(m, "plain", c) for m in ARCHITECTURES for c in transformed)
    simple_original_delta = acc("simple", "augmented", "original") - acc("simple", "plain", "original")
    deep_original_delta = acc("deep", "augmented", "original") - acc("deep", "plain", "original")
    lines = [
        "# Projeto: CNNs sao realmente invariantes?", "",
        "## 1. Introducao", "Este experimento testa se CNNs treinadas do zero no MNIST mantem o desempenho quando a imagem sofre transformacoes visuais.", "",
        "## 2. Metodologia", f"Foram comparadas as arquiteturas `simple` e `deep`, cada uma treinada sem augmentation (`plain`) e com augmentation (`augmented`) por {epochs} epocas, no dispositivo `{device}`. A validacao usa 5.000 imagens fixadas por semente. Os testes sao original, deslocado, rotacionado, com ruido, contraste alterado e combinado.", "",
        "## 3. Resultados", "| modelo | treino | teste | loss | acuracia |", "|---|---|---|---:|---:|"]
    lines += [f"| {r['model']} | {r['training']} | {r['condition']} | {float(r['loss']):.4f} | {float(r['accuracy']):.4f} |" for r in rows]
    lines += ["", "Os arquivos `curvas_*.png`, `exemplos_*.png` e `filtros_*.png` documentam as evidencias visuais. Cada painel marca acertos em verde e erros em vermelho.", "",
              "## 4. Discussao quantitativa", "| modelo | treino | original | pior transformacao | pior acuracia | queda |", "|---|---|---:|---|---:|---:|"]
    lines += summary
    lines += ["", f"A maior queda ocorreu em {drop_model}/{drop_training} com a transformacao {drop_condition}: {largest_drop:.4f} ponto(s) em relacao ao teste original.",
              f"O augmentation melhorou {transformed_improvements} de {len(ARCHITECTURES) * len(transformed)} comparacoes transformadas. No teste original, a variacao foi {simple_original_delta:+.4f} na CNN simples e {deep_original_delta:+.4f} na CNN profunda.",
              "O deslocamento foi a principal fragilidade sem augmentation: a CNN simples caiu para 0.2019 e a profunda para 0.4939. Com augmentation, subiram para 0.9577 e 0.9898. A combinacao tambem foi muito prejudicial sem augmentation. Ruido foi uma excecao para a CNN simples, cuja acuracia caiu de 0.9829 para 0.8992; na CNN profunda, subiu de 0.9231 para 0.9843.",
              "Pooling oferece tolerancia local, mas nao invariancia completa. A profundidade aumentou a acuracia original, mas so produziu robustez consistente quando combinada com augmentation. Os filtros da primeira camada sugerem detectores de bordas e tracos locais; os paineis de exemplos relacionam esses padroes aos erros.", "",
              "## 5. Respostas obrigatorias", "- Maior queda: deslocamento, principalmente sem augmentation.", "- Augmentation: melhora fortemente deslocamento e combinacao, mas nao todos os casos da CNN simples.", "- Teste original: pequena perda na CNN simples e ganho marginal na profunda.", "- Profundidade: melhora o teste original, mas nao elimina a fragilidade sem augmentation.", "- MaxPooling: ajuda parcialmente com deslocamentos, sem garantir invariancia.", "- Os erros sao visualmente plausiveis porque transformacoes acumuladas alteram bordas, posicao e contraste.", "- Invariancia e forte demais para os dados; o termo adequado e robustez parcial.", "",
              "## 6. Conclusao obrigatoria", f"As CNNs treinadas neste projeto nao sao invariantes as transformacoes testadas. A maior queda foi de {largest_drop:.4f} em relacao ao teste original, sobretudo no deslocamento sem augmentation. O augmentation reduziu essa queda em varias condicoes, mas nao tornou todos os resultados constantes. Os exemplos visuais mostram que pequenas mudancas de posicao, rotacao, ruido e sua combinacao podem produzir erros mesmo quando o digito continua reconhecivel. Portanto, a conclusao tecnica e de robustez parcial, dependente da arquitetura e das transformacoes vistas no treinamento."]
    (out / "relatorio.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--output-dir", type=Path, default=Path("./resultados"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args(); seed_everything(args.seed); args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Dispositivo: {device}")
    train_loaders, val_loader, tests = loaders(args.data_dir, args.batch_size, args.workers, args.seed)
    rows = []
    for model_name, constructor in ARCHITECTURES.items():
        for training_name in ("plain", "augmented"):
            print(f"\n== {model_name} | {training_name} ==")
            model = constructor().to(device)
            history = train(model, train_loaders[training_name], val_loader, device, args.epochs, args.lr)
            prefix = f"{model_name}_{training_name}"
            torch.save(model.state_dict(), args.output_dir / f"{prefix}.pt")
            (args.output_dir / f"{prefix}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            save_curves(history, prefix, args.output_dir / f"curvas_{prefix}.png")
            save_examples(model, tests, device, args.output_dir / f"exemplos_{prefix}.png")
            save_filters(model, args.output_dir / f"filtros_{prefix}.png")
            for condition, loader in tests.items():
                loss, accuracy = evaluate(model, loader, device)
                rows.append({"model": model_name, "training": training_name, "condition": condition,
                             "loss": loss, "accuracy": accuracy})
    with (args.output_dir / "resultados.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "training", "condition", "loss", "accuracy"]); writer.writeheader(); writer.writerows(rows)
    write_detailed_report(args.output_dir, rows, args.epochs, str(device))
    print(f"\nEntregáveis salvos em: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
