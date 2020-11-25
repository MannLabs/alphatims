if __name__ == "__main__":
    import alphatims.gui
    import alphatims.utils
    import multiprocessing
    multiprocessing.freeze_support()
    alphatims.utils.set_logger()
    alphatims.gui.run()
