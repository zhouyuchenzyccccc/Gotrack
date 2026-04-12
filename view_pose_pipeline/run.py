from __future__ import annotations

from .config import parse_args


def main():
    args = parse_args()
    from .viewer import run_multiview_viewer, run_single_view_viewer

    if args.raw_data_dir is not None:
        run_multiview_viewer(args)
    else:
        run_single_view_viewer(args)


if __name__ == "__main__":
    main()
