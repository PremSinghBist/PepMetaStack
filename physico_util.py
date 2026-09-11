import numpy as np

EISENBERG = {
    "A": 0.62, "C": 0.29, "D": -1.50, "E": -1.50, "F": 1.19,
    "G": 0.48, "H": -0.40, "I": 1.38, "K": -1.50, "L": 1.06,
    "M": 0.64, "N": -0.78, "P": -1.52, "Q": -0.85, "R": -2.53,
    "S": -0.18, "T": -0.05, "V": 1.08, "W": 0.81, "Y": 0.26
}

CHARGE = {
    "D": -1,
    "E": -1,
    "K": 1,
    "R": 1,
    "H": 0
}


def softmax(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x)       # numerical stability |  Subtracting the maximum value from the input array before applying the exponential function helps prevent numerical overflow. When the input values are large, taking the exponential can result in very large numbers that exceed the floating-point range, leading to infinity or NaN values. By subtracting the maximum value, we ensure that the largest value in the array becomes 0, and all other values become negative, which keeps the exponentials within a manageable range and maintains numerical stability.
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def get_domain_attention(sequence):
    """
    Returns a domain-guided attention vector for a peptide sequence.

    Score for each residue:
        score = hydrophobicity + charge

    Returns:
        attention: shape (sequence_length,)
    """

    scores = []

    for aa in sequence.upper():
        hydrophobicity = EISENBERG[aa]
        charge = CHARGE.get(aa, 0)

        score = hydrophobicity + charge
        scores.append(score)

    return softmax(scores)

if __name__ == "__main__":
    sequence = "ACDKLR"

    attention = get_domain_attention(sequence)

    print(attention)
    print(attention.shape)
    print(attention.sum())