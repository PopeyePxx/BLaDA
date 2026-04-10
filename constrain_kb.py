import base64
from openai import OpenAI
import os
import re
import json
import parse
import numpy as np
import time
from datetime import datetime
from string import Template
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


class ConstraintGenerator:
    def __init__(self, config):
        self.config = config
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        self.base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), './vlm_query')
        with open(os.path.join(self.base_dir, 'prompt_kb.txt'), 'r') as f:
            self.prompt_template = f.read()

    def _load_knowledge_prior(self, max_lines=171):
            triplet_path = '/home/hun/code/GraspSplats-main/all_trip.txt'
            prior_lines = []

            if os.path.exists(triplet_path):
                with open(triplet_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 3:
                            continue
                        task, relation, value = parts
                        if relation == 'whichcomponent':
                            prior_lines.append(f"- In task '{task}', the graspable component is '{value}'.")
                        elif relation == 'whichforce':
                            prior_lines.append(f"- In task '{task}', the required grasp force is '{value}'.")
                        elif relation == 'whichfinger':
                            prior_lines.append(f"- In task '{task}', the involved fingers are '{value}'.")
                        elif relation == 'whichgrasptype':
                            prior_lines.append(f"- In task '{task}', the grasp type is '{value}'.")

            return "\n".join(prior_lines[:max_lines])

    def _build_prompt(self, instruction):
            # knowledge_prior = self._load_knowledge_prior()
            prompt_template = Template(self.prompt_template)
            prompt_text = prompt_template.substitute(instruction=instruction, knowledge_prior=0) # knowledge_prior=knowledge_prior

            # Optionally save prompt for inspection
            os.makedirs(self.task_dir, exist_ok=True)
            with open(os.path.join(self.task_dir, 'prompt_kb.txt'), 'w') as f:
                f.write(prompt_text)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_text
                        }
                    ]
                }
            ]
            return messages

    def _parse_and_save_constraints(self, output, save_dir, metadata=None):
        """
        Parse the JSON output from the language model (containing finger roles and target regions)
        and generate constraint functions for robotic grasping. Each constraint function takes the
        specific finger's end_effector (e.g., thumb_end_effector) and keypoints, using point cloud
        segmentation to map target regions to keypoints. Include control instructions (joint index, force)
        based on finger and role. Save the constraints to files in save_dir.

        Args:
            output (str): JSON string containing finger roles and target regions, e.g.,
                          {"thumb": {"role": "press", "target_region": "handle top"}, ...}
            save_dir (str): Directory to save constraint files
            metadata (dict, optional): Additional data, e.g., point cloud or keypoints
        """
        json_pattern = r'```json\s*(\{[\s\S]*?\})\s*```'  # Match ```json {...} ```
        fallback_pattern = r'\{[\s\S]*\}'  # Match {...} greedily

        json_str = None
        json_match = re.search(json_pattern, output)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(fallback_pattern, output)
            if json_match:
                json_str = json_match.group(0)

        if not json_str:
            print("Error: No JSON block found in output")
            print(f"Raw output: {output}")
            return
        try:
            finger_roles = json.loads(json_str)
            # Validate JSON structure
            required_keys = ["thumb", "index", "middle", "ring", "pinky"]
            if not all(key in finger_roles for key in required_keys):
                raise ValueError(f"Invalid JSON: Missing required fingers {required_keys}")
            for finger, info in finger_roles.items():
                if not all(k in info for k in ["role", "target_region"]):
                    raise ValueError(f"Invalid JSON: Missing role or target_region for {finger}")
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse JSON: {e}")
            print(f"Extracted JSON string: {json_str}")
            print(f"Raw output: {output}")
            return
        except ValueError as e:
            print(f"Error: {e}")
            return

        # Define finger to joint index mapping (example, adjust as needed)
        finger_to_joint = {
            "thumb": 0,
            "index": 1,
            "middle": 2,
            "ring": 3,
            "pinky": 4
        }

        # Define target forces for each role (in Newtons, adjustable)
        role_to_force = {
            "press": 5.0,  # High force for pressing
            "control": 2.0,  # Moderate force for precise control
            "support": 1.0,  # Low force for stability
            "not used": 0.0  # No force
        }

        # Assume clip_segmeter and point cloud are available (via self or metadata)
        clip_segmeter = getattr(self, "clip_segmeter", None)
        point_cloud = metadata.get("point_cloud", np.zeros((1, 3))) if metadata else np.zeros((1, 3))
        keypoints = metadata.get("keypoints", point_cloud[:1]) if metadata else point_cloud[:1]

        # Generate constraint functions based on finger roles
        constraints = []
        keypoint_indices = {}  # Map finger to keypoint index
        for finger, info in finger_roles.items():
            if info["role"] != "not used":
                # Map target_region to keypoint using point cloud segmentation
                keypoint_idx = 0  # Default placeholder
                if clip_segmeter:
                    try:
                        region_similarity = clip_segmeter.compute_similarity_one(info["target_region"], level="subpart")
                        keypoint_idx = np.argmax(region_similarity)
                    except Exception as e:
                        print(f"Warning: Failed to map {info['target_region']} to keypoint: {e}")
                        keypoint_idx = 0  # Fallback

                keypoint_indices[finger] = keypoint_idx

                # Generate constraint function for specific finger's end_effector
                constraint_code = f"""
    def stage_subgoal_{finger}_constraint({finger}_end_effector, keypoints):
        \"\"\"Align the {finger} (joint {finger_to_joint[finger]}) with {info['target_region']} (role: {info['role']}).

        Control instructions:
        - Finger: {finger} (joint index: {finger_to_joint[finger]})
        - Initial angle: 0.0 (radians, adjust dynamically)
        - Initial force: {role_to_force[info['role']]} N (based on role: {info['role']})

        Cost includes:
        - Distance between {finger}_end_effector and target region (keypoint {keypoint_idx})
        - Force deviation based on role
        \"\"\"
        # Distance cost: Align {finger}_end_effector with target region
        distance_cost = np.linalg.norm({finger}_end_effector - keypoints[{keypoint_idx}])

        # Force cost: Deviation from target force
        target_force = {role_to_force[info['role']]}  # Target force in Newtons
        current_force = 0.0  # Placeholder: Replace with force feedback for {finger}
        force_cost = abs(target_force - current_force)

        # Combine costs (weights adjustable)
        cost = distance_cost + 0.1 * force_cost
        return cost
    """
                constraints.append(constraint_code)

        # Group constraints (all under stage1_subgoal for simplicity)
        groupings = {"stage_subgoal": [f"stage_subgoal_{finger}" for finger in finger_roles if
                                        finger_roles[finger]["role"] != "not used"]}

        # Save constraints to files
        for key in groupings:
            if groupings[key]:  # Only save if there are constraints
                with open(os.path.join(save_dir, f"{key}_constraints.txt"), "w") as f:
                    for i, constraint in enumerate(constraints):
                        f.write(constraint + "\n\n")
                        f.write(f"# Function name: {groupings[key][i]}\n\n")
        print(f"Constraints saved to {save_dir}")

    def _parse_other_metadata(self, output):
        data_dict = dict()
        # find num_stages
        num_stages_template = "num_stages = {num_stages}"
        for line in output.split("\n"):
            num_stages = parse.parse(num_stages_template, line)
            if num_stages is not None:
                break
        if num_stages is None:
            raise ValueError("num_stages not found in output")
        data_dict['num_stages'] = int(num_stages['num_stages'])
        # find grasp_keypoints
        grasp_keypoints_template = "grasp_keypoints = {grasp_keypoints}"
        for line in output.split("\n"):
            grasp_keypoints = parse.parse(grasp_keypoints_template, line)
            if grasp_keypoints is not None:
                break
        if grasp_keypoints is None:
            raise ValueError("grasp_keypoints not found in output")
        # convert into list of ints
        grasp_keypoints = grasp_keypoints['grasp_keypoints'].replace("[", "").replace("]", "").split(",")
        grasp_keypoints = [int(x.strip()) for x in grasp_keypoints]
        data_dict['grasp_keypoints'] = grasp_keypoints
        # find release_keypoints
        release_keypoints_template = "release_keypoints = {release_keypoints}"
        for line in output.split("\n"):
            release_keypoints = parse.parse(release_keypoints_template, line)
            if release_keypoints is not None:
                break
        if release_keypoints is None:
            raise ValueError("release_keypoints not found in output")
        # convert into list of ints
        release_keypoints = release_keypoints['release_keypoints'].replace("[", "").replace("]", "").split(",")
        # release_keypoints = [int(x.strip()) for x in release_keypoints]
        release_keypoints = [int(x.split()[0].strip()) for x in release_keypoints if x.split()[0].strip().isdigit()]
        data_dict['release_keypoints'] = release_keypoints
        return data_dict

    def _parse_output(self, output):
        result = {"g^a": "", "g^t": "", "g^f": "", "g^r": ""}
        for line in output.split("\n"):
            if ":" in line:
                key, val = line.split(":", 1)
                result[key.strip()] = val.strip()
        return result

    def generate(self, instruction):
        """
        Args:
            instruction (str): instruction for the query
        Returns:
            save_dir (str): directory where the constraints
        """
        # create a directory for the task

        fname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + instruction.lower().replace(" ", "_")
        self.task_dir = os.path.join(self.base_dir, fname)
        os.makedirs(self.task_dir, exist_ok=True)

        # build prompt
        messages = self._build_prompt(instruction)
        # stream back the response
        stream = self.client.chat.completions.create(model=self.config['model'],
                                                     messages=messages,
                                                     temperature=self.config['temperature'],
                                                     max_tokens=self.config['max_tokens'],
                                                     stream=True,
                                                     timeout=60
                                                     )
        output = ""
        start = time.time()
        for chunk in stream:
            print(f'[{time.time() - start:.2f}s] Querying OpenAI API...', end='\r')
            if chunk.choices[0].delta.content is not None:
                output += chunk.choices[0].delta.content
        print(f'[{time.time() - start:.2f}s] Querying OpenAI API...Done')
        # save raw output
        with open(os.path.join(self.task_dir, 'output_raw.txt'), 'w') as f:
            f.write(output)
        # parse and save constraints
        # self._parse_and_save_constraints(output, self.task_dir)
        # return self.task_dir
        # -----yf-------#
        result = self._parse_output(output)
        return result
        # -----yf-------#