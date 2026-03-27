import qwen_pgd_attack
import torch

attacker = qwen_pgd_attack.Qwen3VLPGD(
    model_path="Qwen/Qwen3-VL-4B-Instruct",
    device="cuda:0",
    dtype=torch.float32,
    vision_only=False,
)

video = "data/QA_Scenes_500/0a8dee95c4ac4ac59a43af56da6e589f/CAM_FRONT.mp4"

result = attacker.attack_video_qa(
    video_path=video,
    question="There is a car that is to the back right of the bus; is it the same status as the truck?",
    answer="yes",
    eps=8/255,
    alpha=1/255,
    num_iter=100,
    fps=1.0,
    verbose=True,
)

print(result)