"""
Span Boundary Objective

- this was used instead of NSP together with SMO
- should this be tried together with SMO and a sentence-level objective?
"""
from pprint import pprint

def get_span_boundaries(span_positions):
    boundary_dict = {}

    for i, position in enumerate(span_positions):
        boundary_dict[position] = {}
        sub = 1
        current = position
        while i - sub >= 0:
            previous = span_positions[i - sub]
            if previous == current - 1:
                current = previous
            else:
                # current does not belong to span anymore
                boundary_dict[position]["left"] = current - 1
                break
            sub += 1
        if "left" not in boundary_dict.get(position):
            boundary_dict[position]["left"] = current - 1

        add = 1
        current = position
        while i + add < len(span_positions):
            next = span_positions[i + add]
            if next == current + 1:
                current = next
            else:
                # current does not belong to span anymore
                boundary_dict[position]["right"] = current + 1
                break
            add += 1
        if "right" not in boundary_dict.get(position):
            boundary_dict[position]["right"] = current + 1

    return boundary_dict


if __name__ == "__main__":
    span_positions = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 42, 80, 81, 82, 83, 84, 85, 86, 87, 88, 150, 151, 152,
                      153, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 222, 249, 250, 251, 252, 274, 275, 303,
                      304, 305, 310, 311]

    boundary_dict = get_span_boundaries(span_positions)
    pprint(boundary_dict)
