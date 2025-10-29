# Copyright (c) Tencent Inc. All rights reserved.
import itertools
from typing import List, Sequence, Tuple
import torch
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm
from mmengine.model import BaseModule
from mmyolo.registry import MODELS
from mmdet.utils import OptMultiConfig, ConfigType
from transformers import (AutoTokenizer, AutoModel, CLIPTextConfig)
from transformers import CLIPTextModelWithProjection as CLIPTP


@MODELS.register_module()
class HuggingVisionBackbone(BaseModule):

    def __init__(self,
                 model_name: str,
                 out_indices: Sequence[int] = (0, 1, 2, 3),
                 norm_eval: bool = True,
                 frozen_modules: Sequence[str] = (),
                 init_cfg: OptMultiConfig = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.norm_eval = norm_eval
        self.frozen_modules = frozen_modules
        self.image_out_indices = out_indices
        self.model = AutoModel.from_pretrained(model_name)

        self._freeze_modules()

    def forward(self, image: Tensor) -> Tuple[Tensor]:
        encoded_dict = self.image_model(pixel_values=image,
                                        output_hidden_states=True)
        hidden_states = encoded_dict.hidden_states
        img_feats = encoded_dict.get('reshaped_hidden_states', hidden_states)
        img_feats = [img_feats[i] for i in self.image_out_indices]
        return tuple(img_feats)

    def _freeze_modules(self):
        for name, module in self.model.named_modules():
            for frozen_name in self.frozen_modules:
                if name.startswith(frozen_name):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
                    break

    def train(self, mode=True):
        super().train(mode)
        self._freeze_modules()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


@MODELS.register_module()
class HuggingCLIPLanguageBackbone(BaseModule):

    def __init__(self,
                 model_name: str,
                 frozen_modules: Sequence[str] = (),
                 dropout: float = 0.0,
                 training_use_cache: bool = False,
                 init_cfg: OptMultiConfig = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.frozen_modules = frozen_modules
        self.training_use_cache = training_use_cache
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        clip_config = CLIPTextConfig.from_pretrained(model_name,
                                                     attention_dropout=dropout)
        self.model = CLIPTP.from_pretrained(model_name, config=clip_config)
        self._freeze_modules()
    def forward_tokenizer(self, texts):
        if not hasattr(self, 'text'):
            text = list(itertools.chain(*texts))
            text = self.tokenizer(text=text, return_tensors='pt', padding=True)
            self.text = text.to(device=self.model.device)
        return self.text

    def forward(self, text: List[List[str]]) -> Tensor:
        num_per_batch = [len(t) for t in text]
        assert max(num_per_batch) == min(num_per_batch), (
            'number of sequences not equal in batch')
        text = list(itertools.chain(*text))
        text = self.tokenizer(text=text, return_tensors='pt', padding=True)
        text = text.to(device=self.model.device)
        txt_outputs = self.model(**text)
        txt_feats = txt_outputs.text_embeds
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        txt_feats = txt_feats.reshape(-1, num_per_batch[0],
                                      txt_feats.shape[-1])
        return txt_feats

    def _freeze_modules(self):

        if len(self.frozen_modules) == 0:
            # not freeze
            return
        if self.frozen_modules[0] == "all":
            self.model.eval()
            for _, module in self.model.named_modules():
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
            return
        for name, module in self.model.named_modules():
            for frozen_name in self.frozen_modules:
                if name.startswith(frozen_name):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
                    break

    def train(self, mode=True):
        super().train(mode)
        self._freeze_modules()
        
@MODELS.register_module()
class HuggingCLIPLanguageBackboneV2(BaseModule):
 
    def __init__(
        self,
        model_name: str,
        frozen_modules: Sequence[str] = (),
        dropout: float = 0.0,
        training_use_cache: bool = False,
        init_cfg: OptMultiConfig = None
    ) -> None:
        super().__init__(init_cfg=init_cfg)
 
        self.frozen_modules = frozen_modules
        self.training_use_cache = training_use_cache
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        clip_config = CLIPTextConfig.from_pretrained(model_name, attention_dropout=dropout)
        self.model = CLIPTP.from_pretrained(model_name, config=clip_config)
 
        self._freeze_modules()
 
    def forward(self, text: List[List[str]] = None) -> Tensor:

        if text is None:
            if self.training:
                # text = [
                #     ["A tank is a tracked combat vehicle with a rotating turret.<br>A tank is equipped with a long-barrel smoothbore gun mounted centrally.<br>A tank boasts a low-profile, compact design that minimizes target exposure on the battlefield."],
                #     ["An armoredcar is a combat vehicle with either wheels or tracks and a reinforced hull.<br>An armoredcar is equipped with a turret-mounted machine gun for fire support in combat operations.<br>An armoredcar features a large, boxy structure designed to maximize interior space and crew protection."],
                #     ["A militarytruck is a heavy-duty vehicle built with a reinforced chassis and a rugged exterior.<br>A militarytruck features high ground clearance, oversized rugged tires, and a spacious cargo area for secure equipment transport."]
                # ]
                text = [
                    ["A tank is a tracked combat vehicle with a rotating turret.<br>A tank features a long-barrel smoothbore gun and thick composite armor.<br>A tank is low-profile design minimizes exposure on the battlefield."],
                    ["An armored car is a combat vehicle with either wheels or tracks and a reinforced hull.<br>An armored car often includes a turret-mounted machine gun for fire support.<br>An armored car has a large, boxy design that provides interior space and crew protection."],
                    ["A military truck is a heavy-duty vehicle with a reinforced chassis and rugged exterior.<br>A military truck has oversized tires, high ground clearance, and a spacious cargo area.<br>A military truck is designed to transport troops, equipment, and supplies across tough terrain."]
                ]

            else:
                # text = [
                #     ["A K2 tank is a main battle tank with modular composite armor and an angular turret.<br>A K2 tank features an integrated a long-barrel smoothbore gun."],
                #     ["A T80 tank is heavily armored battle tank with a low-profile turret and a smoothbore gun.<br>A T80 tank is powered by a gas turbine engine."],
                #     ["A K200 armored car is boxy, fully tracked armored personnel carrier with a high-profile structure and a rear troop hatch.<br>A K200 armored car equipped with a turret-mounted machine gun for self-defense and side armor for additional protection."],
                #     ["A BMP3 armored car is low-profile amphibious infantry fighting vehicle with a flat, boxy turret and a centrally mounted main gun.<br>A BMP3 armored car equipped with a cannon and coaxial machine guns, providing multi-role combat capability."],
                #     ["A K311 military truck is equipped with an extended cargo bed and reinforced side panels for secure equipment storage.<br>A K311 military truck designed with a high-ground clearance chassis and oversized rugged tires."]
                # ]
                text = [
                    ["A tank is a tracked combat vehicle with a rotating turret.<br>A tank features a long-barrel smoothbore gun and thick composite armor.<br>A tank is low-profile design minimizes exposure on the battlefield."],
                    ["An armored car is a combat vehicle with either wheels or tracks and a reinforced hull.<br>An armored car often includes a turret-mounted machine gun for fire support.<br>An armored car has a large, boxy design that provides interior space and crew protection."],
                    ["A military truck is a heavy-duty vehicle with a reinforced chassis and rugged exterior.<br>A military truck has oversized tires, high ground clearance, and a spacious cargo area.<br>A military truck is designed to transport troops, equipment, and supplies across tough terrain."]
                ]
                # text = [
                #     ["The T80 tank is a Russian battle tank with a low-profile turret and composite armor.<br>T80 tank is powered by a gas turbine engine, offering high speed and acceleration.<br>T80 tank is equipped with a 125mm smoothbore gun for superior firepower."],
                #     ["The K2 tank is a modern South Korean main battle tank with modular composite armor.<br>K2 tank has an angular turret and a 120mm smoothbore gun.<br>K2 tank is designed for high mobility and strong firepower on diverse terrains."],
                #     [""],
                #     ["The BMP3 armored car is a low-profile, tracked infantry fighting vehicle.<br>It has a flat turret with a 100mm cannon and coaxial machine guns.<br>BMP3 armored car is amphibious and supports multi-role combat operations."],
                #     ["The K200 armored car is a fully tracked armored personnel carrier from South Korea.<br>K200 armored car has a boxy structure, rear troop hatch, and a turret-mounted machine gun.<br>K200 armoreed car provides crew protection with side armor and mobility over rough terrain."],
                #     [""],
                #     ["The K311 military truck with a high-ground clearance chassis.<br>K311 military truck features an extended cargo bed and rugged tires for off-road transport.<br>K311 military truck is used for carrying equipment and troops in various missions."]
                # ]


        # 나머지는 동일
        split_text = []
        for b_idx, t_list in enumerate(text):
            per_class_splitted = []
            for c_idx, t_str in enumerate(t_list):
                splitted = t_str.split("<br>")
                splitted = [s.strip() for s in splitted if s.strip()]
                per_class_splitted.append(splitted)
            split_text.append(per_class_splitted)

        num_per_batch = [len(t_list) for t_list in text]
        assert max(num_per_batch) == min(num_per_batch), (
            'number of sequences (classes) not equal in batch'
        )

        flatten_splitted = []
        index_map = []
        for b_idx, per_class_splitted in enumerate(split_text):
            for c_idx, lines in enumerate(per_class_splitted):
                for l in lines:
                    flatten_splitted.append(l)
                    index_map.append((b_idx, c_idx))

        tokenized = self.tokenizer(
            text=flatten_splitted,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        tokenized = tokenized.to(device=self.model.device)

        txt_outputs = self.model(**tokenized)
        txt_feats = txt_outputs.text_embeds
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)

        accum = {}
        for (b_idx, c_idx), feat in zip(index_map, txt_feats):
            if (b_idx, c_idx) not in accum:
                accum[(b_idx, c_idx)] = []
            accum[(b_idx, c_idx)].append(feat)

        batch_size = len(text)
        num_classes = num_per_batch[0]
        emb_dim = txt_feats.shape[-1]

        out_feats = torch.zeros(batch_size, num_classes, emb_dim, device=self.model.device)
        for (b_idx, c_idx), feats_list in accum.items():
            feats_tensor = torch.stack(feats_list, dim=0)
            mean_feat = feats_tensor.mean(dim=0)
            out_feats[b_idx, c_idx] = mean_feat

        return out_feats
 
    def _freeze_modules(self):
        if len(self.frozen_modules) == 0:
            return
        if self.frozen_modules[0] == "all":
            self.model.eval()
            for _, module in self.model.named_modules():
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
            return
        for name, module in self.model.named_modules():
            for frozen_name in self.frozen_modules:
                if name.startswith(frozen_name):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
                    break
 
    def train(self, mode=True):
        super().train(mode)
        self._freeze_modules()


# @MODELS.register_module()
# class HuggingCLIPLanguageBackboneV2(BaseModule):
 
#     def __init__(
#         self,
#         model_name: str,
#         frozen_modules: Sequence[str] = (),
#         dropout: float = 0.0,
#         training_use_cache: bool = False,
#         init_cfg: OptMultiConfig = None
#     ) -> None:
#         super().__init__(init_cfg=init_cfg)
 
#         self.frozen_modules = frozen_modules
#         self.training_use_cache = training_use_cache
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         clip_config = CLIPTextConfig.from_pretrained(model_name, attention_dropout=dropout)
#         self.model = CLIPTP.from_pretrained(model_name, config=clip_config)
 
#         self._freeze_modules()
 
#     def forward(self, text: List[List[str]]) -> Tensor:
#         """
#         text의 구조 예시:
#         [
#           [ "클래스1_a<br>클래스1_b", "클래스1_c<br>클래스1_d" ],
#           [ "클래스2_a<br>클래스2_b", "클래스2_c<br>클래스2_d" ],
#           ...
#         ]
 
#         - 바깥쪽 리스트의 길이는 "배치 크기" (batch_size).
#         - 안쪽 리스트의 길이는 "클래스(또는 문장) 수" (num_per_batch_item).
#         - 각 요소(문자열) 안에 <br>로 여러 문장이 연결되어 있음.
 
#         목표:
#         1) 각 문자열을 <br> 기준으로 분리하여 여러 문장으로 만들기
#         2) 같은 배치 아이템(같은 클래스)에 속한 문장들을 임베딩 후 평균
#         3) 최종적으로 (batch_size, num_per_batch_item, emb_dim) 형태의 텐서 반환
#         4) 어떤 문장이 어떤 클래스(혹은 어떤 idx)에 속하는지 인덱싱 정보도 관리
#         """
 
#         # -------------------------------------------------
#         # 1) <br> 분리: 각 배치 아이템 내에 있는 문자열들에서 <br>을 기준으로 문장을 나눔
#         # -------------------------------------------------
#         # split_text[b][c] = ["문장1", "문장2", ...]
#         split_text = []
#         # 각 배치 아이템(b)에 대해
#         for b_idx, t_list in enumerate(text):
#             # t_list: ["클래스1_a<br>클래스1_b", "클래스1_c<br>..."] 등
#             per_class_splitted = []
#             for c_idx, t_str in enumerate(t_list):
#                 splitted = t_str.split("<br>")
#                 # 공백 제거 후 빈 문자열('')은 제외
#                 splitted = [s.strip() for s in splitted if s.strip()]
#                 per_class_splitted.append(splitted)
#             split_text.append(per_class_splitted)
       
#         # -------------------------------------------------
#         # 2) 클래스(혹은 sub-item)별로 몇 개 문장이 생겼는지 기록
#         # -------------------------------------------------
#         # num_per_batch[b] = len(t_list) (각 배치 아이템 내에 "클래스"가 몇 개인지)
#         num_per_batch = [len(t_list) for t_list in text]
#         # 위 assert는 "모든 배치 아이템의 클래스 수가 동일해야 한다"는 가정
#         assert max(num_per_batch) == min(num_per_batch), (
#             'number of sequences (classes) not equal in batch'
#         )
#         # 실제로 <br> 분리를 했기 때문에, 각 클래스(혹은 sub-item)에 문장들이 여러 개 생김
#         # split_text[b][c] = ["문장1", "문장2", ...]
 
#         # -------------------------------------------------
#         # 3) 전체 문장을 Flatten 처리하여 한 번에 토크나이징
#         # -------------------------------------------------
#         # flatten_splitted: (전체 문장 목록)
#         # index_map: 어느 (b_idx, c_idx)에 속하는 문장인지 기록
#         flatten_splitted = []
#         index_map = []  # [(b_idx, c_idx), ...]
#         for b_idx, per_class_splitted in enumerate(split_text):
#             for c_idx, lines in enumerate(per_class_splitted):
#                 for l in lines:
#                     flatten_splitted.append(l)
#                     index_map.append((b_idx, c_idx))
 
#         # -------------------------------------------------
#         # 4) 토크나이징 + 임베딩
#         # -------------------------------------------------
#         tokenized = self.tokenizer(
#             text=flatten_splitted,
#             return_tensors='pt',
#             padding=True,
#             truncation=True
#         )
#         tokenized = tokenized.to(device=self.model.device)
 
#         txt_outputs = self.model(**tokenized)
#         txt_feats = txt_outputs.text_embeds  # (N_total, emb_dim)
#         # L2 정규화
#         txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
 
#         # -------------------------------------------------
#         # 5) (b_idx, c_idx)별 문장 임베딩을 평균
#         # -------------------------------------------------
#         # 임시로 결과를 저장할 딕셔너리: {(b_idx, c_idx): [임베딩들]}
#         accum = {}
#         for (b_idx, c_idx), feat in zip(index_map, txt_feats):
#             if (b_idx, c_idx) not in accum:
#                 accum[(b_idx, c_idx)] = []
#             accum[(b_idx, c_idx)].append(feat)
 
#         # 이제 각 (b_idx, c_idx)에 대해 평균 계산
#         # 최종 shape: (batch_size, num_class_per_batch, emb_dim)
#         # 미리 batch_size와 num_classes는 구해둠
#         batch_size = len(text)
#         num_classes = num_per_batch[0]  # 모두 같다고 가정
#         emb_dim = txt_feats.shape[-1]
       
#         # 결과 텐서 생성
#         # (b_idx, c_idx)에 해당하는 평균 임베딩을 채움
#         out_feats = torch.zeros(batch_size, num_classes, emb_dim, device=self.model.device)
       
#         for (b_idx, c_idx), feats_list in accum.items():
#             feats_tensor = torch.stack(feats_list, dim=0)  # (num_lines, emb_dim)
#             mean_feat = feats_tensor.mean(dim=0)           # (emb_dim,)
#             out_feats[b_idx, c_idx] = mean_feat
 
#         # out_feats shape = [B, N, emb_dim]
#         return out_feats
 
#     def _freeze_modules(self):
#         if len(self.frozen_modules) == 0:
#             return
#         if self.frozen_modules[0] == "all":
#             self.model.eval()
#             for _, module in self.model.named_modules():
#                 module.eval()
#                 for param in module.parameters():
#                     param.requires_grad = False
#             return
#         for name, module in self.model.named_modules():
#             for frozen_name in self.frozen_modules:
#                 if name.startswith(frozen_name):
#                     module.eval()
#                     for param in module.parameters():
#                         param.requires_grad = False
#                     break
 
#     def train(self, mode=True):
#         super().train(mode)
#         self._freeze_modules()
        
# [
#     ["BMPBMD features a fully tracked chassis with an amphibious hull, allowing for operations in water and rough terrain."],
#     ["BMPBMD is equipped with an autocannon and smoke grenade launchers on its turret, enhancing firepower and survivability."],
#     ["BMPBMD has a sloped armor design that improves protection while maintaining a lightweight structure for increased mobility."],
  
#     ["BTR features an 8x8 wheeled chassis, providing high mobility on various terrains, including rough and off-road conditions."],
#     ["BTR is equipped with a turret-mounted autocannon and additional hatches for crew members, enhancing firepower and operational flexibility."],
#     ["BTR has a sloped armored hull with reinforced plating, offering protection against small arms fire and shrapnel while maintaining speed and maneuverability."],
    
#     ["KAMAZ features a robust 6x6 wheeled chassis, providing high mobility and durability across rough terrains and off-road conditions."],
#     ["KAMAZ is designed as a multi-purpose military truck capable of transporting troops, cargo, and equipment efficiently in various operational environments."],
#     ["KAMAZ has a reinforced cab with optional armor protection, ensuring crew safety while maintaining reliability in harsh conditions."],

#     ["MTLB features a fully tracked chassis with an amphibious capability, allowing it to traverse water obstacles and rough terrain efficiently."],
#     ["MTLB is designed as a multi-purpose armored vehicle, capable of transporting troops, cargo, and serving as a platform for various weapon systems."],
#     ["MTLB has a low-profile armored hull that provides protection against small arms fire and shrapnel while maintaining high mobility and adaptability."],
    
#     ["RLS features an advanced radar-based detection system mounted on an armored vehicle, providing early warning and situational awareness."],
#     ["RLS integrates high-mobility capabilities with cutting-edge radar technology, making it essential for tracking and identifying threats in real-time."],
#     ["RLS has an armored chassis with reinforced protection, ensuring operational reliability in hostile environments and combat zones."],

#     ["RSZO features a heavy-duty 8x8 wheeled chassis, providing high mobility and stability for rapid artillery deployment."],
#     ["RSZO is equipped with multiple rocket launch tubes capable of firing salvos of unguided rockets, delivering high-impact area saturation."],
#     ["RSZO has an armored cab to protect the crew from small arms fire and shrapnel while operating in combat zones."],
    
#     ["SAU features a tracked chassis with an amphibious hull, allowing for operations in both water and rugged terrain."],
#     ["SAU is equipped with a large-caliber self-propelled gun mounted on a low-profile turret, providing long-range fire support."],
#     ["SAU has a lightweight armored structure that enhances mobility while offering protection against small arms fire and shrapnel."],

#     ["URAL features a rugged 6x6 wheeled chassis, designed for high mobility and durability across various terrains, including off-road conditions."],
#     ["URAL is a versatile military truck capable of transporting troops, cargo, and equipment, supporting a wide range of logistical operations."],
#     ["URAL has a reinforced cab with optional armor protection, ensuring crew safety while maintaining reliability in harsh environmental conditions."],

#     ["ZRK features a wheeled armored chassis, providing high mobility and maneuverability for rapid deployment in various terrains."],
#     ["ZRK is equipped with multiple surface-to-air missiles and an advanced targeting system, designed to intercept and neutralize aerial threats."],
#     ["ZRK has an angular armored hull with reinforced plating, offering protection for the crew while maintaining agility in combat operations."],
    
#     ["T80 features a fully tracked chassis with advanced composite armor, providing enhanced protection against various threats."],
#     ["T80 is equipped with a 125mm smoothbore gun capable of firing APFSDS, HEAT, and guided missiles for superior firepower."],
#     ["T80 has a gas turbine engine, offering high speed and mobility on diverse terrains while maintaining rapid acceleration capabilities."]
# ]

@MODELS.register_module()
class PseudoLanguageBackbone(BaseModule):
    """Pseudo Language Backbone
    Args:
        text_embed_path (str): path to the text embedding file
    """

    def __init__(self,
                 text_embed_path: str = "",
                 test_embed_path: str = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)
        # {text:embed}
        self.text_embed = torch.load(text_embed_path, map_location='cpu')
        if test_embed_path is None:
            self.test_embed = self.text_embed
        else:
            self.test_embed = torch.load(test_embed_path)
        self.register_buffer("buff", torch.zeros([
            1,
        ]))

    def forward_cache(self, text: List[List[str]]) -> Tensor:
        if not hasattr(self, "cache"):
            self.cache = self.forward_text(text)
        return self.cache

    def forward(self, text: List[List[str]]) -> Tensor:
        if self.training:
            return self.forward_text(text)
        else:
            return self.forward_cache(text)

    def forward_text(self, text: List[List[str]]) -> Tensor:
        num_per_batch = [len(t) for t in text]
        assert max(num_per_batch) == min(num_per_batch), (
            'number of sequences not equal in batch')
        text = list(itertools.chain(*text))
        if self.training:
            text_embed_dict = self.text_embed
        else:
            text_embed_dict = self.test_embed
        text_embeds = torch.stack(
            [text_embed_dict[x.split("/")[0]] for x in text])
        # requires no grad and force to float
        text_embeds = text_embeds.to(
            self.buff.device).requires_grad_(False).float()
        text_embeds = text_embeds.reshape(-1, num_per_batch[0],
                                          text_embeds.shape[-1])
        return text_embeds


# @MODELS.register_module()
# class MultiModalYOLOBackbone(BaseModule):

#     def __init__(self,
#                  image_model: ConfigType,
#                  text_model: ConfigType,
#                  frozen_stages: int = -1,
#                  with_text_model: bool = True,
#                  init_cfg: OptMultiConfig = None) -> None:
#         super().__init__(init_cfg)
#         self.with_text_model = with_text_model
#         self.image_model = MODELS.build(image_model)
#         if self.with_text_model:
#             self.text_model = MODELS.build(text_model)
#         else:
#             self.text_model = None
#         self.frozen_stages = frozen_stages
#         self._freeze_stages()

#     def _freeze_stages(self):
#         """Freeze the parameters of the specified stage so that they are no
#         longer updated."""
#         if self.frozen_stages >= 0:
#             for i in range(self.frozen_stages + 1):
#                 m = getattr(self.image_model, self.image_model.layers[i])
#                 m.eval()
#                 for param in m.parameters():
#                     param.requires_grad = False

#     def train(self, mode: bool = True):
#         """Convert the model into training mode while keep normalization layer
#         frozen."""
#         super().train(mode)
#         self._freeze_stages()

#     def forward(self, image: Tensor,
#                 text: List[List[str]]) -> Tuple[Tuple[Tensor], Tensor]:
#         img_feats = self.image_model(image)
#         if self.with_text_model:
#             txt_feats = self.text_model(text)
#             return img_feats, txt_feats
#         else:
#             return img_feats, None

#     def forward_text(self, text: List[List[str]]) -> Tensor:
#         assert self.with_text_model, "forward_text() requires a text model"
#         txt_feats = self.text_model(text)
#         return txt_feats

#     def forward_image(self, image: Tensor) -> Tuple[Tensor]:
#         return self.image_model(image)

from mmengine.model import BaseModule
from mmengine.registry import MODELS
from torch import Tensor
from typing import List, Tuple, Any

@MODELS.register_module()
class MultiModalYOLOBackbone(BaseModule):
    def __init__(
        self,
        image_model: Any,
        text_model_det: Any = None,
        text_model_grounding: Any = None,
        text_model: Any = None,
        frozen_stages: int = -1,
        init_cfg: Any = None
    ) -> None:
        super().__init__(init_cfg)
        # Build image backbone
        self.image_model = MODELS.build(image_model)
        # Build specialized text backbones
        self.text_model_det = MODELS.build(text_model_det) if text_model_det else None
        self.text_model_grounding = MODELS.build(text_model_grounding) if text_model_grounding else None
        # Legacy single text_model (used if no det/grounding provided)
        self.text_model = None
        if text_model and not (text_model_det or text_model_grounding):
            self.text_model = MODELS.build(text_model)
        # Freeze specified stages
        self.frozen_stages = frozen_stages
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                layer_name = self.image_model.layers[i]
                m = getattr(self.image_model, layer_name)
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_stages()

    def forward(
        self,
        image: Tensor,
        text: List[List[str]],
        mode: str = 'det'
    ) -> Tuple[Tuple[Tensor], Tensor]:
        """
        Args:
            image: input image tensor
            text: list of text samples
            mode: 'det' or 'grounding' to select the text backbone
        Returns:
            img_feats: tuple of image feature tensors
            txt_feats: text feature tensor (or None)
        """
        img_feats = self.image_model(image)
        txt_feats = None
        if mode == 'det' and self.text_model_det:
            txt_feats = self.text_model_det(text)
        elif mode == 'grounding' and self.text_model_grounding:
            txt_feats = self.text_model_grounding(text)
        elif self.text_model:
            txt_feats = self.text_model(text)
        return img_feats, txt_feats

    def forward_text(self, text: List[List[str]]) -> Tensor:
        if self.text_model_grounding:
            return self.text_model_grounding(text)
        if self.text_model_det:
            return self.text_model_det(text)
        assert self.text_model, "No text model available"
        return self.text_model(text)

    def forward_image(self, image: Tensor) -> Tuple[Tensor]:
        return self.image_model(image)