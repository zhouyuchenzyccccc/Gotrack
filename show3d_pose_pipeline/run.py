from __future__ import annotations

from .config import parse_args


def main():
    args = parse_args()
    from .controller import run_show3d_sequence

    run_show3d_sequence(args)


if __name__ == "__main__":
    main()

