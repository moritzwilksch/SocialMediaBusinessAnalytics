#%%
from typing import Tuple
from sklearn.metrics import accuracy_score, mean_absolute_error
from rich.console import Console
root_path = "../"
VADER_THRESH = 0.05
c = Console(highlight=False)

def eval_regression(ytrue, ypred, print=True) -> Tuple[int, int]:
    """ Returns mae, acc given ytrue & ypred. """
    binary_ytrue = (ytrue > 0).astype('int8')  # use ">" because a buy signal should only be generated for positive returns
    binary_ypred = (ypred > 0).astype('int8')  # use ">" because a buy signal should only be generated for positive returns

    acc = accuracy_score(binary_ytrue, binary_ypred)
    mae = mean_absolute_error(ytrue, ypred)

    c.print(f"Accuracy = {acc:.3f}", style="white on blue")
    c.print(f"MAE = {mae:.4f}", style="white on blue")

    return mae, acc

#%%


def eval_classification(ytrue, ypred, print=True) -> float:
    """ Returns acc given binary ytrue & ypred. """
    acc = accuracy_score(ytrue, ypred)

    c.print(f"Accuracy = {acc:.3f}", style="white on blue")

    return acc
