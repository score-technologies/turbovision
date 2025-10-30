from enum import Enum
from pathlib import Path
from zipfile import ZipFile
from json import loads, load
from logging import getLogger

from numpy import ndarray
from pydantic import BaseModel
from SoccerNet.Downloader import SoccerNetDownloader
from cv2 import imread, VideoWriter, VideoWriter_fourcc

from scorevision.vlm_pipeline.utils.response_models import (
    ShirtColor,
    TEAM1_SHIRT_COLOUR,
    TEAM2_SHIRT_COLOUR,
)
from scorevision.utils.settings import get_settings

logger = getLogger(__name__)


# =====TURBOVISION ANNOTATIONS======
class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    cls_id: int
    conf: float


class TVFrameResult(BaseModel):
    frame_id: int
    boxes: list[BoundingBox]
    keypoints: list[tuple[int, int]]


class TurbovisionSoccerObjects(Enum):
    BALL = "ball"
    GOALIE = "goalkeeper"
    PLAYER = "player"
    REFEREE = "referee"
    PLAYER_TEAM_1 = "team 1"
    PLAYER_TEAM_2 = "team 2"


OBJECT_ID_REVERSE_LOOKUP = {
    TurbovisionSoccerObjects.BALL: 0,
    TurbovisionSoccerObjects.GOALIE: 1,
    TurbovisionSoccerObjects.PLAYER: 2,
    TurbovisionSoccerObjects.REFEREE: 3,
    TurbovisionSoccerObjects.PLAYER_TEAM_1: 6,
    TurbovisionSoccerObjects.PLAYER_TEAM_2: 7,
}

# =====SOCCERNET ANNOTATIONS======

ID_TO_SHIRT_COLOUR = {
    0: ShirtColor.GREY,
    1: ShirtColor.PINK,
    3: ShirtColor.BROWN,
    6: TEAM1_SHIRT_COLOUR,
    7: TEAM2_SHIRT_COLOUR,
}


class PointPositionInLine(Enum):
    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"
    # TOP_INTERSECTION = "top-intersection"
    # BOTTOM_INTERSECTION = "bottom-intersection"


# NOTE: having redundancy from keypoints appearing in more than one line is good since not all lines appear fully in each frame and where both appear, we can check intersections, etc
KEYPOINTS_FROM_SOCCERNET_LINES = {
    1: {
        "Side line top": {"point": PointPositionInLine.LEFT},
        "Side line left": {"point": PointPositionInLine.TOP},
    },
    2: {
        "Big rect. left top": {"point": PointPositionInLine.LEFT},
        "Side line left": None,
    },
    3: {
        "Small rect. left top": {"point": PointPositionInLine.LEFT},
        "Side line left": None,
    },
    4: {
        "Small rect. left bottom": {"point": PointPositionInLine.LEFT},
        "Side line left": None,
    },
    5: {
        "Big rect. left bottom": {"point": PointPositionInLine.LEFT},
        "Side line left": None,
    },
    6: {
        "Side line left": {"point": PointPositionInLine.BOTTOM},
        "Side line bottom": {"point": PointPositionInLine.LEFT},
    },
    7: {
        "Small rect. left top": {"point": PointPositionInLine.RIGHT},
        "Small rect. left main": {"point": PointPositionInLine.TOP},
    },
    8: {
        "Small rect. left bottom": {"point": PointPositionInLine.RIGHT},
        "Small rect. left main": {"point": PointPositionInLine.BOTTOM},
    },
    9: {},  # NOTE: some keypoints have no corresponding lines and so we cannot get those
    10: {
        "Big rect. left top": {"point": PointPositionInLine.RIGHT},
        "Big rect. left main": {"point": PointPositionInLine.TOP},
    },
    11: {
        "Big rect. left main": None,
        "Circle left": {"point": PointPositionInLine.TOP},
    },
    12: {
        "Big rect. left main": None,
        "Circle left": {"point": PointPositionInLine.BOTTOM},
    },
    13: {
        "Big rect. left bottom": {"point": PointPositionInLine.RIGHT},
        "Big rect. left main": {"point": PointPositionInLine.BOTTOM},
    },
    14: {"Side line top": None, "Middle line": {"point": PointPositionInLine.TOP}},
    15: {
        "Middle line": None,
        "Circle central": {"point": PointPositionInLine.TOP},  # TODO:.TOP_INTERSECTION}
    },
    16: {
        "Middle line": None,
        "Circle central": {
            "point": PointPositionInLine.BOTTOM
        },  # TODO:BOTTOM_INTERSECTION}
    },
    17: {
        "Middle line": {"point": PointPositionInLine.BOTTOM},
        "Side line bottom": None,
    },
    18: {
        "Big rect. right top": {"point": PointPositionInLine.LEFT},
        "Big rect. right main": {"point": PointPositionInLine.TOP},
    },
    19: {
        "Circle right": {"point": PointPositionInLine.TOP},
        "Big rect. right main": None,
    },
    20: {
        "Circle right": {"point": PointPositionInLine.BOTTOM},
        "Big rect. right main": None,
    },
    21: {
        "Big rect. right bottom": {"point": PointPositionInLine.LEFT},
        "Big rect. right main": {"point": PointPositionInLine.BOTTOM},
    },
    22: {},
    23: {
        "Small rect. right top": {"point": PointPositionInLine.LEFT},
        "Small rect. right main": {"point": PointPositionInLine.TOP},
    },
    24: {
        "Small rect. right bottom": {"point": PointPositionInLine.LEFT},
        "Small rect. right main": {"point": PointPositionInLine.BOTTOM},
    },
    25: {
        "Side line top": {"point": PointPositionInLine.RIGHT},
        "Side line right": {"point": PointPositionInLine.TOP},
    },
    26: {
        "Big rect. right top": {"point": PointPositionInLine.RIGHT},
        "Side line right": None,
    },
    27: {
        "Small rect. right top": {"point": PointPositionInLine.RIGHT},
        "Side line right": None,
    },
    28: {
        "Small rect. right bottom": {"point": PointPositionInLine.RIGHT},
        "Side line right": None,
    },
    29: {
        "Big rect. right bottom": {"point": PointPositionInLine.RIGHT},
        "Side line right": None,
    },
    30: {
        "Side line bottom": {"point": PointPositionInLine.RIGHT},
        "Side line right": {"point": PointPositionInLine.BOTTOM},
    },
    31: {},
    32: {},
}


class Metadata(BaseModel):
    version: str
    game_id: str
    id: str
    num_tracklets: str
    action_position: str
    action_class: str
    visibility: str
    game_time_start: str
    game_time_stop: str
    clip_start: str
    clip_stop: str
    name: str
    im_dir: str
    frame_rate: int
    seq_length: int
    im_ext: str


class ImageData(BaseModel):
    is_labeled: bool
    image_id: str
    file_name: str
    height: int
    width: int
    has_labeled_person: bool
    has_labeled_pitch: bool
    has_labeled_camera: bool
    ignore_regions_y: list
    ignore_regions_x: list


class AttributeData(BaseModel):
    role: str
    jersey: str | None = None
    team: str | None = None


class BBoxImageData(BaseModel):
    x: int
    y: int
    x_center: float
    y_center: float
    w: int
    h: int


class BBoxPitchData(BaseModel):
    x_bottom_left: float
    y_bottom_left: float
    x_bottom_right: float
    y_bottom_right: float
    x_bottom_middle: float
    y_bottom_middle: float


class Coordinate(BaseModel):
    x: float
    y: float


class Line(BaseModel):
    coordinates: list[Coordinate]

    @property
    def top(self) -> Coordinate:
        return min(self.coordinates, key=lambda coord: coord.y)

    @property
    def bottom(self) -> Coordinate:
        return max(self.coordinates, key=lambda coord: coord.y)

    @property
    def left(self) -> Coordinate:
        return min(self.coordinates, key=lambda coord: coord.x)

    @property
    def right(self) -> Coordinate:
        return max(self.coordinates, key=lambda coord: coord.x)


class Annotation(BaseModel):
    id: str
    image_id: str
    track_id: int | None = None
    supercategory: str | None = None
    category_id: int | None = None
    attributes: AttributeData | None = None
    bbox_image: BBoxImageData | None = None
    bbox_pitch: BBoxPitchData | None = None
    bbox_pitch_raw: BBoxPitchData | None = None
    lines: dict[str, list[Coordinate]] | None = None


class Category(BaseModel):
    supercategory: str
    id: int
    name: str
    lines: list[str] | None = None


class SoccernetGamestateDataset(BaseModel):
    info: Metadata
    images: list[ImageData]
    annotations: list[Annotation]
    categories: list[Category]


SOCCERNET_CATEGORY_ID_LOOKUP = {
    1: TurbovisionSoccerObjects.PLAYER,
    2: TurbovisionSoccerObjects.GOALIE,
    3: TurbovisionSoccerObjects.REFEREE,
    4: TurbovisionSoccerObjects.BALL,
}

SOCCERNET_TEAM_LOOKUP = {
    "right": TurbovisionSoccerObjects.PLAYER_TEAM_1,
    "left": TurbovisionSoccerObjects.PLAYER_TEAM_2,
}


def generate_video(
    frames: list[ndarray], path: Path, frame_height: int, frame_width: int
) -> None:
    _fourcc = VideoWriter_fourcc(*"mp4v")
    out = VideoWriter(str(path), _fourcc, 25.0, (frame_width, frame_height))
    for frame in frames:
        out.write(frame)
    out.release()
    logger.info(f"Saved video: {path}")


def extract_keypoint_coordinate_from_line(
    keypoint_index: int, lines: dict[str, Line]
) -> Coordinate:
    coord = Coordinate(x=0.0, y=0.0)
    required_lines = KEYPOINTS_FROM_SOCCERNET_LINES[keypoint_index]
    if not any(required_lines):
        logger.info("No possible lines to ever get this keypoint. Skipping")
        return coord

    if not all(required_line in lines for required_line in required_lines):
        logger.info(
            f"Keypoint {keypoint_index} requires {len(required_lines)} specific lines to be visible for reliable extraction but some are missing. Skipping"
        )
        return coord

    for line_name, coord_position in required_lines.items():
        if coord_position is None:
            continue
        line = lines[line_name]
        coord = getattr(line, coord_position["point"].value)
        logger.info(f"Keypoint {keypoint_index} = {coord}")
    return coord


def get_bounding_boxes(annotations: list[Annotation]) -> list[BoundingBox]:
    bboxes = []
    for annotation in annotations:
        if not annotation.attributes:
            logger.debug("Not a bounding box object. Skipping")
            continue
        if annotation.attributes and annotation.attributes.team:
            object_type = SOCCERNET_TEAM_LOOKUP[annotation.attributes.team]
            bbox = BoundingBox(
                x1=annotation.bbox_image.x,
                y1=annotation.bbox_image.y,
                x2=annotation.bbox_image.x + annotation.bbox_image.w,
                y2=annotation.bbox_image.y + annotation.bbox_image.h,
                cls_id=OBJECT_ID_REVERSE_LOOKUP[object_type],
                conf=1.0,
            )

        else:
            object_type = SOCCERNET_CATEGORY_ID_LOOKUP.get(annotation.category_id)
            if object_type is None:
                logger.debug("Not a bounding box object of interest. Skipping")
                continue
            bbox = BoundingBox(
                x1=annotation.bbox_image.x,
                y1=annotation.bbox_image.y,
                x2=annotation.bbox_image.x + annotation.bbox_image.w,
                y2=annotation.bbox_image.y + annotation.bbox_image.h,
                cls_id=OBJECT_ID_REVERSE_LOOKUP[object_type],
                conf=1.0,
            )
        bboxes.append(bbox)
    return bboxes


def get_lines(annotations: list[Annotation]) -> dict[str, Line]:
    for annotation in annotations:
        if annotation.lines:
            return {
                key: Line(coordinates=values)
                for key, values in annotation.lines.items()
            }
    return {}


def transform_lines_into_keypoints(
    lines: dict[str, Line], frame_width: int, frame_height: int
) -> list[tuple[int, int]]:
    keypoints = [(0, 0) for _ in range(len(KEYPOINTS_FROM_SOCCERNET_LINES))]
    if any(lines):
        for keypoint_index in KEYPOINTS_FROM_SOCCERNET_LINES:
            coord = extract_keypoint_coordinate_from_line(
                keypoint_index=keypoint_index,
                lines=lines,
            )
            keypoints[keypoint_index - 1] = (
                int(coord.x * frame_width),
                int(coord.y * frame_height),
            )
    return keypoints


def download_raw_dataset(
    save_directory: str, dataset_name: str, dataset_split: str
) -> None:
    settings = get_settings()
    downloader = SoccerNetDownloader(LocalDirectory=save_directory)
    if settings.SOCCERNET_TOKEN.get_secret_value() is None:
        raise Exception("SOCCERNET_TOKEN not set")
    downloader.password = settings.SOCCERNET_TOKEN.get_secret_value()
    downloader.downloadDataTask(task=dataset_name, split=[dataset_split])
    logger.info(f"Dataset downloaded to {save_directory}")


def unzip_raw_dataset(path_zipfile: Path) -> None:
    with ZipFile(path_zipfile) as zip_ref:
        zip_ref.extractall(path_zipfile.with_suffix(""))
    path_zipfile.unlink()
    logger.info(f"{path_zipfile} unzipped and deleted")


def load_annotation_for_video(path: Path) -> SoccernetGamestateDataset:
    with path.open() as f:
        return SoccernetGamestateDataset(**load(f))


def load_annotations_for_videos(
    dataset_directory: Path,
    annotations_path: Path,
    video_path: Path,
    annotation_json_filename: str,
    image_subdirectory_name: str,
) -> dict[str, list[TVFrameResult]]:
    dataset = {}
    for i, path in enumerate(
        dataset_directory.glob(f"*/{annotation_json_filename}.json")
    ):
        videoname = path.parent.stem
        path_gt = annotations_path / f"{videoname}.jsonl"
        path_video = video_path / f"{videoname}.mp4"

        if path_gt.exists():
            logger.info(f"{path_gt} already exists. Skipping processing step...")
            with path_gt.open() as f:
                turbovision_data = [TVFrameResult(**loads(line)) for line in f]
        else:
            soccernet_data = load_annotation_for_video(path=path)
            logger.info(
                f"Processing video {videoname} with {len(soccernet_data.images)} frames"
            )
            frames = []
            turbovision_data = []
            for image_data in soccernet_data.images:
                tv_annotation, frame = (
                    convert_soccernet_to_turbovision_annotation_format(
                        image_directory=path.parent / image_subdirectory_name,
                        image_data=image_data,
                        annotations=soccernet_data.annotations,
                    )
                )
                turbovision_data.append(tv_annotation)
                frames.append(frame)

            with path_gt.open("w") as f:
                for tv_frame_result in turbovision_data:
                    f.write(tv_frame_result.json() + "\n")

            if not path_video.exists():
                h = soccernet_data.images[0].height
                w = soccernet_data.images[0].width
                if not path_video.exists():
                    generate_video(
                        frames=frames,
                        path=path_video,
                        frame_height=h,
                        frame_width=w,
                    )
        dataset[videoname] = turbovision_data
    return dataset


def convert_soccernet_to_turbovision_annotation_format(
    image_directory: Path,
    image_data: ImageData,
    annotations: list[Annotation],
) -> tuple[TVFrameResult, ndarray]:
    path_image = image_directory / image_data.file_name
    annotations_in_frame = [
        annotation
        for annotation in annotations
        if annotation.image_id == image_data.image_id
    ]
    logger.info(f"{len(annotations_in_frame)} annotations found for frame {path_image}")
    bboxes = get_bounding_boxes(annotations=annotations_in_frame)
    logger.info(f"{len(bboxes)} bboxes formatted")
    lines = get_lines(annotations=annotations_in_frame)
    logger.info(f"{len(lines)} lines detected in frame")
    keypoints = transform_lines_into_keypoints(
        lines=lines, frame_width=image_data.width, frame_height=image_data.height
    )
    logger.info(f"lines converted to {len(keypoints)} keypoints")

    annotation = TVFrameResult(
        frame_id=int(Path(image_data.file_name).stem), boxes=bboxes, keypoints=keypoints
    )
    unannotated_frame = None
    if path_image.exists():
        unannotated_frame = imread(str(path_image))
    return annotation, unannotated_frame
