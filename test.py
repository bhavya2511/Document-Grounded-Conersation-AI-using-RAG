try:
    from paddleocr import PPStructureV3
    print("Import successful!")
    engine = PPStructureV3()
    print("Instantiation successful!")
except Exception as e:
    import traceback
    traceback.print_exc()

