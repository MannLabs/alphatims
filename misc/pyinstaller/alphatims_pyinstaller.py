if __name__ == "__main__":
    import alphatims.utils
    log_file_name = alphatims.utils.set_logger(log_file_name="")
    threads = alphatims.utils.set_threads(-1)
    alphatims.utils.show_platform_info()
    alphatims.utils.show_python_info()
    alphatims.utils.check_github_version()
    import logging
    logging.info("Running one-click GUI with parameters:")
    logging.info(f"log_file_name - {log_file_name}")
    logging.info(f"threads       - {threads}")
    logging.info("")
    logging.info("Loading GUI...")
    import alphatims.gui
    import multiprocessing
    multiprocessing.freeze_support()
    alphatims.gui.run()
