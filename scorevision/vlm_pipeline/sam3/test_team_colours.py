


if __name__ == "__main__":
    from asyncio import run
    from cv2 import imread
    from logging import basicConfig, INFO

    from vlm_pipeline.sam3.detect_team_colours import sam3_extract_shirt_colours

    basicConfig(level=INFO)
    image = imread("vlm_pipeline/sam3/football.jpg")
    colours =  run(sam3_extract_shirt_colours(image=image))
    print(colours)
 

