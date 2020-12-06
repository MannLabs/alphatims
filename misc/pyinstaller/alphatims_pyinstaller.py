if __name__ == "__main__":
    import alphatims.utils
    log_file_name = alphatims.utils.set_logger(
        log_file_name=alphatims.utils.INTERFACE_PARAMETERS["log_file"][
            "default"
        ]
    )
    threads = alphatims.utils.set_threads(
        alphatims.utils.INTERFACE_PARAMETERS["threads"]["default"]
    )
    alphatims.utils.show_platform_info()
    alphatims.utils.show_python_info()
    import logging
    logging.info("Running one-click GUI with parameters:")
    logging.info(f"log_file_name - {log_file_name}")
    logging.info(f"threads       - {threads}")
    logging.info("")
    logging.info("Loading GUI..")
    import alphatims.gui
    import multiprocessing
    multiprocessing.freeze_support()
    alphatims.gui.run()
