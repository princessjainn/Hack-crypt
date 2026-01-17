# Awesome Deepfakes Detection![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)

![from internet](assets/cover.jpg)
<small>*(image from internet)*</small>

A collection list of Deepfakes Detection related datasets, tools, papers, and code. If this list helps you on your research, a star will be my pleasure :)

If you want to contribute to this list, welcome to send me a pull request or contact me :)

This repository only collects papers related to Deepfake Detection. If you are also interested in Deepfakes generation, please refer to: [Awesome Deepfakes](https://github.com/Daisy-Zhang/Awesome-Deepfakes).

## Contents

- [Datasets](#datasets)
# Truefy Backend (FastAPI + PyTorch)

Deepfake forensics backend powering image, video, and audio analysis. Provides a simple FastAPI API that accepts media uploads and returns verdict, probabilities, and confidence.

## Features
- FastAPI service with CORS enabled for local frontend
- PyTorch image model with temperature scaling and thresholds
- Video frame extraction with robust averaging
- Audio deepfake inference
- Unified `POST /predict` endpoint returning `FAKE | REAL | UNCERTAIN`

## Tech Stack
- FastAPI, Uvicorn
- PyTorch (`torch`, `torchvision`, `torchaudio`)
- OpenCV, Pillow, NumPy
- Librosa (audio)

## Project Structure
- `backend/app.py`: FastAPI app and endpoints
- `backend/models/image_model.py`: Image classifier
- `backend/video/*`: Frame extraction and temporal analysis
- `backend/audio/*`: Audio inference pipeline
- `backend/preprocessing/image.py`: Image preprocessing
- `models/*.pth`: Pretrained weights
- `uploads/`, `temp_frames/`: Runtime storage

## Setup (Windows)
1. Open a terminal in the `BACKEND` folder.
2. Create and activate a virtual environment:

```powershell
* Do You Really Mean That? Content Driven Audio-Visual Deepfake Dataset and Multimodal Method for Temporal Forgery Localization, *arXiv* 2022: [Paper](https://arxiv.org/abs/2204.06228)
* Lip Sync Matters: A Novel Multimodal Forgery Detector, *APSIPA* 2022: [Paper](https://homepage.iis.sinica.edu.tw/papers/whm/25387-F.pdf)
* Multimodal Forgery Detection Using Ensemble Learning, *APSIPA* 2022: [Paper](https://www.researchgate.net/profile/Ammarah-Hashmi/publication/365603278_Multimodal_Forgery_Detection_Using_Ensemble_Learning/links/6379afc62f4bca7fd075912e/Multimodal-Forgery-Detection-Using-Ensemble-Learning.pdf)

3. Install dependencies:

```powershell
* Deepfake Video Detection Based on Spatial, Spectral, and Temporal Inconsistencies Using Multimodal Deep Learning, *AIPR* 2020: [Paper](https://ieeexplore.ieee.org/abstract/document/9425167/)


4. Verify model files exist in `models/`:
	- `image_deepfake.pth`
	- `video_temporal.pth` (optional for temporal)
	- `audio_deepfake.pth` (optional for audio)

## Run
Start the API with reload for local development:

```powershell



You should see health info on [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health).

## API
- `GET /` → service status
- `GET /health` → device info (`cpu`/`cuda`)
- `POST /predict` (multipart `file`)

Response (example for image):

```json
## Biological Signal

* Local attention and long-distance interaction of rPPG for deepfake detection, *The Visual Computer* 2023: [Paper](https://link.springer.com/article/10.1007/s00371-023-02833-x)
* Benchmarking Joint Face Spoofing and Forgery Detection with Visual and Physiological Cues, *arXiv* 2022: [Paper](https://arxiv.org/abs/2208.05401)
* Visual Representations of Physiological Signals for Fake Video Detection, *arXiv* 2022: [Paper](https://arxiv.org/abs/2207.08380)
* Study of detecting behavioral signatures within DeepFake videos, *arXiv* 2022: [Paper](https://arxiv.org/abs/2208.03561)
* Detecting Deep-Fake Videos from Aural and Oral Dynamics, *CVPR Workshop* 2021: [Paper](https://openaccess.thecvf.com/content/CVPR2021W/WMF/html/Agarwal_Detecting_Deep-Fake_Videos_From_Aural_and_Oral_Dynamics_CVPRW_2021_paper.html)
* Countering Spoof: Towards Detecting Deepfake with Multidimensional Biological Signals, *Security and Communication Networks* 2021: [Paper](https://www.hindawi.com/journals/scn/2021/6626974/)

Verdict policy (from `backend/app.py`):
- Images: `FAKE_THRESHOLD = 0.70` → `FAKE` if fake ≥ 70%, `REAL` if ≤ 30%, otherwise `UNCERTAIN`.
- Videos: trimmed average of frame fake probs → `FAKE/REAL/FAKE` (bias toward `FAKE` for uncertain).
- Audio: `FAKE` if ≥ 75%, `REAL` if ≤ 30%, otherwise `UNCERTAIN`.

## CORS
Frontend connects from `localhost:5173` (Vite). CORS is enabled for:
- `http://localhost:5173`, `http://127.0.0.1:5173`
- `http://localhost:8080`, `http://127.0.0.1:8080`

## Troubleshooting
- If CUDA isn’t available, model runs on CPU.
- Large videos: ensure `temp_frames/` has write permission.
- 415 Unsupported Media Type: send as multipart with `file` field.

## Development Tips
- Adjust `FAKE_THRESHOLD` in `backend/app.py` to tune sensitivity.
- Log device info and predictions via `logging` already configured.

* A Study on Effective Use of BPM Information in Deepfake Detection, *ICTC* 2021: [Paper](https://ieeexplore.ieee.org/abstract/document/9621186/)
* Exposing Deepfake with Pixel-wise Autoregressive and PPG Correlation from Faint Signals, *arXiv* 2021: [Paper](https://arxiv.org/abs/2110.15561)
* FakeCatcher: Detection of Synthetic Portrait Videos using Biological Signals, *TPAMI* 2020: [Paper](https://ieeexplore.ieee.org/abstract/document/9141516/)
* How Do the Hearts of Deep Fakes Beat? Deep Fake Source Detection via Interpreting Residuals with Biological Signals, *IJCB* 2020: [Paper](https://ieeexplore.ieee.org/abstract/document/9304909/)
* DeepFakesON-Phys: DeepFakes Detection based on Heart Rate Estimation, *arXiv* 2020: [Paper](https://arxiv.org/abs/2010.00400) [Github](https://github.com/BiDAlab/DeepFakesON-Phys)
* Predicting Heart Rate Variations of Deepfake Videos using Neural ODE, *ICCV Workshop* 2019: [Paper](https://openaccess.thecvf.com/content_ICCVW_2019/papers/CVPM/Fernandes_Predicting_Heart_Rate_Variations_of_Deepfake_Videos_using_Neural_ODE_ICCVW_2019_paper.pdf)


## Robustness

* LAA-Net: Localized Artifact Attention Network for Quality-Agnostic and Generalizable Deepfake Detection, *CVPR* 2024: [Paper](https://arxiv.org/pdf/2401.13856) [Github](https://github.com/10Ring/LAA-Net)
* Quality-Agnostic Deepfake Detection with Intra-model Collaborative Learning, *ICCV 2023*: [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Le_Quality-Agnostic_Deepfake_Detection_with_Intra-model_Collaborative_Learning_ICCV_2023_paper.pdf)


## Fairness

* Preserving Fairness Generalization in Deepfake Detection, *CVPR* 2024: [Paper](https://arxiv.org/pdf/2402.17229) [Github](https://github.com/Purdue-M2/Fairness-Generalization)
* GBDF: Gender Balanced DeepFake Dataset Towards Fair DeepFake Detection, *ICPR* 2022: [Paper](https://arxiv.org/abs/2207.10246) [Github](https://github.com/aakash4305/GBDF)
* A Comprehensive Analysis of AI Biases in DeepFake Detection With Massively Annotated Databases, *arXiv* 2022: [Paper](https://arxiv.org/abs/2208.05845) [Github](https://github.com/pterhoer/DeepFakeAnnotations)
* An Examination of Fairness of AI Models for Deepfake Detection, *IJCAI* 2021: [Paper](https://www.ijcai.org/proceedings/2021/0079.pdf)



## Fingerprint Watermark

* Responsible Disclosure of Generative Models Using Scalable Fingerprinting, *ICLR* 2022: [Paper](https://openreview.net/forum?id=sOK-zS6WHB) [Github](https://github.com/ningyu1991/ScalableGANFingerprints)
* DeepFake Disrupter: The Detector of DeepFake Is My Friend, *CVPR* 2022: [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_DeepFake_Disrupter_The_Detector_of_DeepFake_Is_My_Friend_CVPR_2022_paper.html)
* FingerprintNet: Synthesized Fingerprints for Generated Image Detection, *ECCV* 2022: [Paper](https://link.springer.com/chapter/10.1007/978-3-031-19781-9_5)
* CMUA-Watermark: A Cross-Model Universal Adversarial Watermark for Combating Deepfakes, *AAAI* 2022: [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/19982) [Github](https://github.com/VDIGPKU/CMUA-Watermark)
* Defeating DeepFakes via Adversarial Visual Reconstruction, *ACM MM* 2022: [Paper](https://dl.acm.org/doi/abs/10.1145/3503161.3547923?casa_token=ZM9dDIwll78AAAAA:BELSycUIPfukaK-ffgIq8bBY7UKm52-gS1yfunR86wwL5uBCFIVtgEeIQnTahZgW1pPGR67rxotieoo)
* DeepFakes for Privacy: Investigating the Effectiveness of State-of-the-Art Privacy-Enhancing Face Obfuscation Methods, *Proceedings of the 2022 International Conference on Advanced Visual Interfaces* 2022: [Paper](https://dl.acm.org/doi/abs/10.1145/3531073.3531125?casa_token=tMrO_mFD_l4AAAAA:CY8GOT2ApoClF-vlCljDRedbdNRljt1S9Xkli4tBsbkThYIQMZwskEg3DRdZXeo0YgpeYxeZ9SmU1Gc)
* Metamorphic Testing-based Adversarial Attack to Fool Deepfake Detectors, *ICPR* 2022: [Paper](https://arxiv.org/abs/2204.08612)
* FaceSigns: Semi-Fragile Neural Watermarks for Media Authentication and Countering Deepfakes, *arXiv* 2022: [Paper](https://arxiv.org/abs/2204.01960) [Github](https://github.com/paarthneekhara/FaceSignsDemo)
* Anti-Forgery: Towards a Stealthy and Robust DeepFake Disruption Attack via Adversarial Perceptual-aware Perturbations, *arXiv* 2022: [Paper](https://arxiv.org/abs/2206.00477) [Github](https://github.com/AbstractTeen/AntiForgery/)
* System Fingerprints Detection for DeepFake Audio: An Initial Dataset and Investigation, *arXiv* 2022: [Paper](https://arxiv.org/abs/2208.10489)
* Artificial Fingerprinting for Generative Models: Rooting Deepfake Attribution in Training Data, *ICCV* 2021: [Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Yu_Artificial_Fingerprinting_for_Generative_Models_Rooting_Deepfake_Attribution_in_Training_ICCV_2021_paper.html) [Github](https://github.com/ningyu1991/ArtificialGANFingerprints)
* FaceGuard: Proactive Deepfake Detection, *CoRR* 2021: [Paper](https://arxiv.org/pdf/2109.05673v1.pdf)


## Identity-Related

* TI2Net: Temporal Identity Inconsistency Network for Deepfake Detection, *WACV* 2023: [Paper](https://openaccess.thecvf.com/content/WACV2023/papers/Liu_TI2Net_Temporal_Identity_Inconsistency_Network_for_Deepfake_Detection_WACV_2023_paper.pdf)
* Protecting Celebrities from DeepFake with Identity Consistency Transformer, *CVPR* 2022: [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Dong_Protecting_Celebrities_From_DeepFake_With_Identity_Consistency_Transformer_CVPR_2022_paper.html) [Github](https://github.com/LightDXY/ICT_DeepFake)
* Protecting World Leader Using Facial Speaking Pattern Against Deepfakes, *IEEE Signal Processing Letters* 2022: [Paper](https://ieeexplore.ieee.org/abstract/document/9882301/)
* Protecting world leaders against deep fakes using facial, gestural, and vocal mannerisms, *Proceedings of the National Academy of Sciences*  2022: [Paper](https://www.pnas.org/doi/abs/10.1073/pnas.2216035119)
* Study of detecting behavioral signatures within DeepFake videos, *arXiv* 2022: [Paper](https://arxiv.org/abs/2208.03561)
* Voice-Face Homogeneity Tells Deepfake, *arXiv* 2022: [Paper](https://arxiv.org/abs/2203.02195)
* Audio-Visual Person-of-Interest DeepFake Detection, *arXiv* 2022: [Paper](https://arxiv.org/abs/2204.03083) [Github](https://github.com/grip-unina/poi-forensics)
* Detecting Deep-Fake Videos from Aural and Oral Dynamics, *CVPR Workshop* 2021: [Paper](https://openaccess.thecvf.com/content/CVPR2021W/WMF/html/Agarwal_Detecting_Deep-Fake_Videos_From_Aural_and_Oral_Dynamics_CVPRW_2021_paper.html)
* ID-Reveal: Identity-aware DeepFake Video Detection, *ICCV* 2021: [Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Cozzolino_ID-Reveal_Identity-Aware_DeepFake_Video_Detection_ICCV_2021_paper.html) [Github](https://github.com/grip-unina/id-reveal)
* This Face Does Not Exist... But It Might Be Yours! Identity Leakage in Generative Models, *WACV* 2021: [Paper](https://openaccess.thecvf.com/content/WACV2021/html/Tinsley_This_Face_Does_Not_Exist..._But_It_Might_Be_Yours_WACV_2021_paper.html)
* An Experimental Evaluation on Deepfake Detection using Deep Face Recognition, *ICCST* 2021: [Paper](https://ieeexplore.ieee.org/abstract/document/9717407/)
* Detecting Deep-Fake Videos from Appearance and Behavior, *WIFS* 2020: [Paper](https://ieeexplore.ieee.org/abstract/document/9360904/)
* Identity-Driven DeepFake Detection, *arXiv* 2020: [Paper](https://arxiv.org/abs/2012.03930)
* Protecting World Leaders Against Deep Fakes, *CVPR Workshop* 2019: [Paper](https://openaccess.thecvf.com/content_CVPRW_2019/papers/Media%20Forensics/Agarwal_Protecting_World_Leaders_Against_Deep_Fakes_CVPRW_2019_paper.pdf)



## Adversarial Attack

* Hiding Faces in Plain Sight: Defending DeepFakes by Disrupting Face Detection, *TDSC* 2025: [Paper](https://ieeexplore.ieee.org/abstract/document/11106399) [Github](https://github.com/OUC-VAS/FacePoison)
* Poisoned Forgery Face: Towards Backdoor Attacks on Face Forgery Detection, *ICLR* 2024: [Paper](https://openreview.net/pdf?id=8iTpB4RNvP) [Github](https://github.com/JWLiang007/PFF)
* Self-supervised Learning of Adversarial Example: Towards Good Generalizations for Deepfake Detection, *CVPR* 2022: [Paper](https://arxiv.org/abs/2203.12208) [Github](https://github.com/liangchen527/SLADD)
* Exploring Frequency Adversarial Attacks for Face Forgery Detection, *CVPR* 2022: [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Jia_Exploring_Frequency_Adversarial_Attacks_for_Face_Forgery_Detection_CVPR_2022_paper.pdf)
* TAFIM: Targeted Adversarial Attacks against Facial Image Manipulations, *ECCV* 2022: [Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740053.pdf) [Github](https://github.com/shivangi-aneja/TAFIM)
* Anti-Forgery: Towards a Stealthy and Robust DeepFake Disruption Attack via Adversarial Perceptual-aware Perturbations, *IJCAI* 2022: [Paper](https://www.ijcai.org/proceedings/2022/0107.pdf) [Github](https://github.com/AbstractTeen/AntiForgery/)
* Investigate Evolutionary Strategies for Black-box Attacks to Deepfake Forensic Systems, *SoICT* 2022: [Paper](https://dl.acm.org/doi/pdf/10.1145/3568562.3568666)
* Evaluating Robustness of Sequence-based Deepfake Detector Models by Adversarial Perturbation, *WDC* 2022: [Paper](https://dl.acm.org/doi/abs/10.1145/3494109.3527194)
* Restricted Black-box Adversarial Attack Against DeepFake Face Swapping, *CoRR* 2022: [Paper](https://arxiv.org/abs/2204.12347)



## Real Scenario

* Contrastive Pseudo Learning for Open-World DeepFake Attribution, *ICCV 2023*: [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Sun_Contrastive_Pseudo_Learning_for_Open-World_DeepFake_Attribution_ICCV_2023_paper.pdf)
* A Continual Deepfake Detection Benchmark: Dataset, Methods, and Essentials, *WACV* 2023: [Paper](https://openaccess.thecvf.com/content/WACV2023/supplemental/Li_A_Continual_Deepfake_WACV_2023_supplemental.pdf)
* Robust Image Forgery Detection Against Transmission Over Online Social Networks, *TIFS* 2022: [Paper](https://ieeexplore.ieee.org/abstract/document/9686650/) [Github](https://github.com/HighwayWu/ImageForensicsOSN)
* Am I a Real or Fake Celebrity? Evaluating Face Recognition and Verification APIs under Deepfake Impersonation Attack, *Proceedings of the ACM Web Conference* 2022: [Paper](https://dl.acm.org/doi/abs/10.1145/3485447.3512212?casa_token=1exG7H-Zf5gAAAAA:bjJAkJKfAP8Ls7ohbQc3PyaTe8s_j_C-8QCca4INNw3eFWxhltDGvsSF7s_D-uVHdsapMJ4II5shIp4) [Github](https://github.com/shahroztariq/Deepfake_Impersonation_Attack)
* DeSI: Deepfake Source Identifier for Social Media, *CVPR Workshop* 2022: [Paper](https://openaccess.thecvf.com/content/CVPR2022W/FaDE-TCV/html/Narayan_DeSI_Deepfake_Source_Identifier_for_Social_Media_CVPRW_2022_paper.html)
* Investigate Evolutionary Strategies for Black-box Attacks to Deepfake Forensic Systems, *SoICT* 2022: [Paper](https://dl.acm.org/doi/pdf/10.1145/3568562.3568666)
* Seeing is Living? Rethinking the Security of Facial Liveness Verification in the Deepfake Era, *CoRR* 2022: [Paper](https://www.usenix.org/conference/usenixsecurity22/presentation/li-changjiang)
* DeFakePro: Decentralized DeepFake Attacks Detection using ENF Authentication, *arXiv* 2022: [Paper](https://arxiv.org/abs/2207.13070)
* DF-Captcha: A Deepfake Captcha for Preventing Fake Calls, *arXiv* 2022: [Paper](https://arxiv.org/abs/2208.08524)
* Practical Deepfake Detection: Vulnerabilities in Global Contexts, *arXiv* 2022: [Paper](https://arxiv.org/abs/2206.09842)
* My Face My Choice: Privacy Enhancing Deepfakes for Social Media Anonymization, *arXiv* 2022: [Paper](https://arxiv.org/abs/2211.01361)
* Preventing DeepFake Attacks on Speaker Authentication by Dynamic Lip Movement Analysis, *TIFS* 2021: [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9298826) [Github](https://github.com/chenzhao-yang/lip-based-anti-spoofing)
* Real, Forged or Deep Fake? Enabling the Ground Truth on the Internet, *IEEE Access* 2021: [Paper](https://ieeexplore.ieee.org/abstract/document/9628088/)
* DeepFake-o-meter: An Open Platform for DeepFake Detection, *SP Workshop* 2021: [Paper](https://arxiv.org/abs/2103.02018)
* Towards Untrusted Social Video Verification to Combat Deepfakes via Face Geometry Consistency, *CVPR Workshop* 2020: [Paper](https://openaccess.thecvf.com/content_CVPRW_2020/html/w39/Tursman_Towards_Untrusted_Social_Video_Verification_to_Combat_Deepfakes_via_Face_CVPRW_2020_paper.html)



## Anomaly Detection

* Self-Supervised Video Forensics by Audio-Visual Anomaly Detection, *arXiv* 2023: [Paper](https://arxiv.org/abs/2301.01767) [Github](https://github.com/cfeng16/audio-visual-forensics)
* Learning Second Order Local Anomaly for General Face Forgery Detection, *CVPR* 2022: [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Fei_Learning_Second_Order_Local_Anomaly_for_General_Face_Forgery_Detection_CVPR_2022_paper.html)
* SeeABLE: Soft Discrepancies and Bounded Contrastive Learning for Exposing Deepfakes, *arXiv* 2022: [Paper](https://arxiv.org/abs/2211.11296)
* Differential Anomaly Detection for Facial Images, *WIFS* 2021: [Paper](https://ieeexplore.ieee.org/abstract/document/9648392/)
* Fighting Deepfakes by Detecting GAN DCT Anomalies, *Journal of Imaging* 2021: [Paper](https://www.mdpi.com/2313-433X/7/8/128)



## Self-Supervised Learning

* Self-Supervised Video Forensics by Audio-Visual Anomaly Detection, *arXiv* 2023: [Paper](https://arxiv.org/abs/2301.01767) [Github](https://github.com/cfeng16/audio-visual-forensics)
* End-to-End Reconstruction-Classification Learning for Face Forgery Detection, *CVPR* 2022: [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Cao_End-to-End_Reconstruction-Classification_Learning_for_Face_Forgery_Detection_CVPR_2022_paper.html)
* Leveraging Real Talking Faces via Self-Supervision for Robust Forgery Detection, *CVPR* 2022: [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Haliassos_Leveraging_Real_Talking_Faces_via_Self-Supervision_for_Robust_Forgery_Detection_CVPR_2022_paper.html)
* UIA-ViT: Unsupervised Inconsistency-Aware Method based on Vision Transformer for Face Forgery Detection, *ECCV* 2022: [Paper](https://link.springer.com/chapter/10.1007/978-3-031-20065-6_23)
* Dual Contrastive Learning for General Face Forgery Detection, *AAAI* 2022: [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/20130)
* Self-supervised Transformer for Deepfake Detection, *arXiv* 2022: [Paper](https://arxiv.org/abs/2203.01265)
* MagDR: Mask-guided Detection and Reconstruction for Defending Deepfakes, *CVPR* 2021: [Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_MagDR_Mask-Guided_Detection_and_Reconstruction_for_Defending_Deepfakes_CVPR_2021_paper.html)
* DeepfakeUCL: Deepfake Detection via Unsupervised Contrastive Learning, *IJCNN* 2021: [Paper](https://arxiv.org/abs/2104.11507)
* Deepfake videos detection using self-supervised decoupling network, *ICME* 2021: [Paper](https://ieeexplore.ieee.org/abstract/document/9428368/)
* Detecting Deep-Fake Videos from Appearance and Behavior, *WIFS* 2020: [Paper](https://ieeexplore.ieee.org/abstract/document/9360904/)



## Source Model Attribution

* Deepfake Network Architecture Attribution, *AAAI* 2022: [Paper](https://aaai-2022.virtualchair.net/poster_aaai4380) [Github](https://github.com/ICTMCG/DNA-Det)
* Model Attribution of Face-swap Deepfake Videos, *ICIP* 2022: [Paper](https://ieeexplore.ieee.org/document/9897972)
* On the Exploitation of Deepfake Model Recognition, *CVPR Workshop* 2022: [Paper](https://openaccess.thecvf.com/content/CVPR2022W/WMF/papers/Guarnera_On_the_Exploitation_of_Deepfake_Model_Recognition_CVPRW_2022_paper.pdf)
* Artificial Fingerprinting for Generative Models: Rooting Deepfake Attribution in Training Data, *ICCV* 2021: [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Yu_Artificial_Fingerprinting_for_Generative_Models_Rooting_Deepfake_Attribution_in_Training_ICCV_2021_paper.pdf) [Github](https://github.com/ningyu1991/ArtificialGANFingerprints)
* Towards Discovery and Attribution of Open-world GAN Generated Images, *ICCV* 2021: [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Girish_Towards_Discovery_and_Attribution_of_Open-World_GAN_Generated_Images_ICCV_2021_paper.pdf)
* Improving Generalization of Deepfake Detection by Training for Attribution, *MMSP* 2021: [Paper](https://ieeexplore.ieee.org/abstract/document/9733468/)
* How Do the Hearts of Deep Fakes Beat? Deep Fake Source Detection via Interpreting Residuals with Biological Signals, *IJCB* 2020: [Paper](https://ieeexplore.ieee.org/abstract/document/9304909/)
* Attributing Fake Images to GANs: Learning and Analyzing GAN Fingerprints, *ICCV* 2019: [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yu_Attributing_Fake_Images_to_GANs_Learning_and_Analyzing_GAN_Fingerprints_ICCV_2019_paper.pdf)
* On Attribution of Deepfakes, *arXiv* 2020: [Paper](https://arxiv.org/pdf/2008.09194.pdf)
* Scalable Fine-grained Generated Image Classification Based on Deep Metric Learning, *CoRR* 2019: [Paper](https://arxiv.org/abs/1912.11082)


## Multiclass Classification

* Three-classification Face Manipulation Detection Using Attention-based Feature Decomposition, *Computers & Security* 2022: [Paper](https://www.sciencedirect.com/science/article/pii/S0167404822004163)
* Forgery-Domain-Supervised Deepfake Detection with Non-Negative Constraint, *IEEE Signal Processing Letters* 2022: [Paper](https://ieeexplore.ieee.org/abstract/document/9839430/)



## Federated Learning

* FedForgery: Generalized Face Forgery Detection with Residual Federated Learning, *arXiv* 2022: [Paper](https://arxiv.org/abs/2210.09563) [Github](https://github.com/GANG370/FedForgery)


## Knowledge Distillation

* Confidence-Calibrated Face Image Forgery Detection with Contrastive Representation Distillation, *ACCV* 2022: [Paper](https://openaccess.thecvf.com/content/ACCV2022/html/Yang_Confidence-Calibrated_Face_Image_Forgery_Detection_with_Contrastive_Representation_Distillation_ACCV_2022_paper.html) [Github](https://github.com/Puning97/CDC_face_forgery_detection)

## Meta-Learning

* Domain General Face Forgery Detection by Learning to Weight, *AAAI* 2021: [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/16367) [Github](https://github.com/skJack/LTW)

## Depth Based

* A guided-based approach for deepfake detection: RGB-depth integration via features fusion, *Pattern Recognition Letters* 2024: [Paper](https://www.sciencedirect.com/science/article/pii/S0167865524000990) [Github](https://github.com/gleporoni/rgbd-depthfake)
* DepthFake: a depth-based strategy for detecting Deepfake videos, *arXiv* 2022: [Paper](https://arxiv.org/abs/2208.11074)
* Exploring Depth Information for Face Manipulation Detection, *arXiv* 2022: [Paper](https://arxiv.org/abs/2212.14230)

