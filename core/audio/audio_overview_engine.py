"""
PPuRI-AI Ultimate - Audio Overview 엔진
NotebookLM 스타일의 대화형 팟캐스트 생성

Features:
- 문서 요약 → 대화형 스크립트 변환
- 한국어 TTS (Edge-TTS, MeloTTS, OpenAI TTS)
- 두 명의 AI 호스트 대화 형식
- 뿌리산업 전문용어 발음 최적화
- OpenRouter LLM 통합
"""

import asyncio
import os
import json
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, AsyncIterator
import logging

logger = logging.getLogger(__name__)


class TTSProvider(Enum):
    """TTS 제공자"""
    EDGE_TTS = "edge_tts"      # Microsoft Edge TTS (무료, 고품질)
    OPENAI_TTS = "openai_tts"  # OpenAI TTS-1/TTS-1-HD
    ELEVENLABS = "elevenlabs"  # ElevenLabs (최고 품질)
    MELO_TTS = "melo_tts"      # MeloTTS (오픈소스, 한국어)
    VITS = "vits"              # 로컬 VITS 모델


class SpeakerRole(Enum):
    """화자 역할"""
    HOST_A = "host_a"     # 메인 호스트 (설명자)
    HOST_B = "host_b"     # 서브 호스트 (질문자)
    EXPERT = "expert"     # 전문가 (선택적)


@dataclass
class DialogueTurn:
    """대화 턴"""
    speaker: SpeakerRole
    text: str
    emotion: str = "neutral"  # neutral, curious, excited, thoughtful
    pause_after_ms: int = 500


@dataclass
class AudioSegment:
    """오디오 세그먼트"""
    speaker: SpeakerRole
    audio_path: str
    duration_seconds: float
    text: str


@dataclass
class AudioOverview:
    """완성된 Audio Overview"""
    title: str
    summary: str
    audio_path: str
    duration_seconds: float
    transcript: List[DialogueTurn]
    created_at: datetime
    source_documents: List[str]


class AudioOverviewEngine:
    """
    NotebookLM 스타일 Audio Overview 생성 엔진

    프로세스:
    1. 문서 분석 및 핵심 내용 추출
    2. 대화형 스크립트 생성 (LLM)
    3. TTS로 음성 합성
    4. 오디오 편집 및 통합
    """

    def __init__(
        self,
        llm_client,
        tts_provider: TTSProvider = TTSProvider.EDGE_TTS,
        output_dir: str = "./audio_outputs"
    ):
        self.llm = llm_client
        self.tts_provider = tts_provider
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 화자별 목소리 설정 (Edge-TTS 기준)
        self.voice_config = {
            SpeakerRole.HOST_A: {
                "voice": "ko-KR-InJoonNeural",    # 남성, 설명 톤
                "rate": "+0%",
                "pitch": "+0Hz"
            },
            SpeakerRole.HOST_B: {
                "voice": "ko-KR-SunHiNeural",     # 여성, 질문 톤
                "rate": "+5%",
                "pitch": "+5Hz"
            },
            SpeakerRole.EXPERT: {
                "voice": "ko-KR-HyunsuNeural",    # 남성, 전문가 톤
                "rate": "-5%",
                "pitch": "-5Hz"
            }
        }

        # 뿌리산업 전문용어 발음 사전
        self.pronunciation_dict = {
            "주조": "주조",
            "금형": "금형",
            "용접": "용접",
            "소성가공": "소성가공",
            "표면처리": "표면처리",
            "열처리": "열처리",
            "다이캐스팅": "다이 캐스팅",
            "TIG": "티그",
            "MIG": "미그",
            "PVD": "피브이디",
            "CVD": "씨브이디",
            "CNC": "씨엔씨",
            "CAD": "캐드",
            "CAM": "캠",
        }

        self._initialized = False

    async def initialize(self) -> bool:
        """엔진 초기화"""
        try:
            # TTS 라이브러리 확인
            if self.tts_provider == TTSProvider.EDGE_TTS:
                import edge_tts
                logger.info("Edge-TTS available")

            self._initialized = True
            return True
        except ImportError as e:
            logger.warning(f"TTS library not available: {e}")
            return False

    async def generate_overview(
        self,
        documents: List[Dict[str, Any]],
        title: str,
        style: str = "conversational",
        duration_target_minutes: int = 10
    ) -> AudioOverview:
        """
        Audio Overview 생성

        Args:
            documents: 소스 문서들
            title: 개요 제목
            style: 스타일 (conversational, educational, technical)
            duration_target_minutes: 목표 길이 (분)
        """
        logger.info(f"Generating Audio Overview: {title}")

        # 1. 문서 내용 요약 및 핵심 추출
        summary = await self._summarize_documents(documents)

        # 2. 대화형 스크립트 생성
        dialogue = await self._generate_dialogue_script(
            summary=summary,
            title=title,
            style=style,
            target_duration=duration_target_minutes
        )

        # 3. TTS로 음성 합성
        audio_segments = await self._synthesize_audio(dialogue)

        # 4. 오디오 통합
        final_audio_path, total_duration = await self._merge_audio_segments(
            segments=audio_segments,
            title=title
        )

        return AudioOverview(
            title=title,
            summary=summary,
            audio_path=final_audio_path,
            duration_seconds=total_duration,
            transcript=dialogue,
            created_at=datetime.now(),
            source_documents=[doc.get("title", "Unknown") for doc in documents]
        )

    async def _summarize_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> str:
        """문서 요약"""
        combined_content = "\n\n---\n\n".join([
            f"제목: {doc.get('title', 'N/A')}\n내용: {doc.get('content', '')[:3000]}"
            for doc in documents
        ])

        prompt = f"""다음 기술 문서들의 핵심 내용을 요약해주세요.
뿌리산업(주조, 금형, 용접, 소성가공, 표면처리, 열처리) 관점에서 중요한 포인트를 강조해주세요.

문서 내용:
{combined_content}

요약 형식:
1. 핵심 주제 (1-2문장)
2. 주요 포인트 (3-5개)
3. 실무 적용 시사점
4. 관련 기술 연결점
"""

        from core.llm import ModelTier
        response = await self.llm.generate(
            prompt=prompt,
            tier=ModelTier.FAST,
            temperature=0.3,
            max_tokens=2000
        )
        return response.content

    async def _generate_dialogue_script(
        self,
        summary: str,
        title: str,
        style: str,
        target_duration: int
    ) -> List[DialogueTurn]:
        """대화형 스크립트 생성"""

        # 대화 턴당 약 30초 = 목표 시간 * 2 턴
        target_turns = target_duration * 2

        style_guide = {
            "conversational": "친근하고 자연스러운 대화체로",
            "educational": "교육적이고 체계적인 설명 형식으로",
            "technical": "전문적이고 깊이 있는 기술 토론 형식으로"
        }

        prompt = f"""당신은 기술 팟캐스트 대본 작가입니다.
다음 요약 내용을 바탕으로 두 명의 호스트가 나누는 자연스러운 대화를 작성해주세요.

제목: {title}
스타일: {style_guide.get(style, '친근한 대화체로')}
목표 대화 턴 수: 약 {target_turns}개

요약 내용:
{summary}

대화 규칙:
1. HOST_A (진행자): 핵심 내용을 설명하고 이야기를 이끌어 갑니다
2. HOST_B (질문자): 청취자를 대신해 궁금한 점을 질문하고 추가 설명을 요청합니다
3. 뿌리산업 전문용어는 처음 등장 시 쉽게 풀어서 설명해주세요
4. 각 대화 턴은 30초 내외로 읽을 수 있는 길이로 (약 80-120자)
5. 자연스러운 추임새, 반응을 포함해주세요 ("네, 맞아요", "흥미롭네요" 등)

출력 형식 (JSON):
[
  {{"speaker": "HOST_A", "text": "대화 내용", "emotion": "neutral"}},
  {{"speaker": "HOST_B", "text": "대화 내용", "emotion": "curious"}},
  ...
]
"""

        from core.llm import ModelTier
        response = await self.llm.generate(
            prompt=prompt,
            tier=ModelTier.REASONING,  # 고품질 대화 생성을 위해 Reasoning 모델 사용
            temperature=0.7,
            max_tokens=4000
        )

        # JSON 파싱
        try:
            # 응답에서 JSON 추출
            content = response.content
            # JSON 블록 찾기
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            dialogue_data = json.loads(content.strip())
            dialogue = []
            for turn in dialogue_data:
                speaker = SpeakerRole.HOST_A if turn.get("speaker") == "HOST_A" else SpeakerRole.HOST_B
                dialogue.append(DialogueTurn(
                    speaker=speaker,
                    text=turn.get("text", ""),
                    emotion=turn.get("emotion", "neutral"),
                    pause_after_ms=800 if turn.get("emotion") == "thoughtful" else 500
                ))
            return dialogue
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse dialogue script: {e}")
            logger.debug(f"Raw response: {response.content[:500]}")
            return []

    async def _synthesize_audio(
        self,
        dialogue: List[DialogueTurn]
    ) -> List[AudioSegment]:
        """TTS 음성 합성"""
        segments = []

        for i, turn in enumerate(dialogue):
            try:
                # 발음 최적화
                text = self._optimize_pronunciation(turn.text)

                # 음성 설정
                voice_config = self.voice_config.get(turn.speaker, self.voice_config[SpeakerRole.HOST_A])

                # TTS 합성
                audio_path = self.output_dir / f"segment_{i:04d}.mp3"

                if self.tts_provider == TTSProvider.EDGE_TTS:
                    await self._synthesize_edge_tts(
                        text=text,
                        voice=voice_config["voice"],
                        rate=voice_config["rate"],
                        pitch=voice_config["pitch"],
                        output_path=str(audio_path)
                    )
                elif self.tts_provider == TTSProvider.MELO_TTS:
                    await self._synthesize_melo_tts(
                        text=text,
                        speaker=turn.speaker,
                        output_path=str(audio_path)
                    )
                elif self.tts_provider == TTSProvider.OPENAI_TTS:
                    await self._synthesize_openai_tts(
                        text=text,
                        voice=voice_config.get("openai_voice", "nova"),
                        output_path=str(audio_path)
                    )

                # 오디오 길이 측정
                duration = await self._get_audio_duration(str(audio_path))

                segments.append(AudioSegment(
                    speaker=turn.speaker,
                    audio_path=str(audio_path),
                    duration_seconds=duration,
                    text=turn.text
                ))

            except Exception as e:
                logger.error(f"Failed to synthesize segment {i}: {e}")
                continue

        return segments

    async def _synthesize_edge_tts(
        self,
        text: str,
        voice: str,
        rate: str,
        pitch: str,
        output_path: str
    ) -> None:
        """Edge-TTS로 음성 합성"""
        try:
            import edge_tts

            communicate = edge_tts.Communicate(
                text=text,
                voice=voice,
                rate=rate,
                pitch=pitch
            )
            await communicate.save(output_path)

        except ImportError:
            logger.error("edge_tts not installed. Run: pip install edge-tts")
            raise

    async def _synthesize_melo_tts(
        self,
        text: str,
        speaker: SpeakerRole,
        output_path: str
    ) -> None:
        """MeloTTS로 음성 합성 (오픈소스 한국어 TTS)"""
        try:
            from melo.api import TTS

            # MeloTTS 모델 (한국어)
            device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
            model = TTS(language="KR", device=device)

            # 화자별 스피커 ID
            speaker_ids = model.hps.data.spk2id
            speaker_id = list(speaker_ids.values())[0]  # 기본 화자

            # 음성 합성
            model.tts_to_file(
                text=text,
                speaker_id=speaker_id,
                output_path=output_path,
                speed=1.0 if speaker == SpeakerRole.HOST_A else 1.1
            )

        except ImportError:
            logger.warning("MeloTTS not available, falling back to Edge-TTS")
            await self._synthesize_edge_tts(
                text=text,
                voice=self.voice_config[speaker]["voice"],
                rate=self.voice_config[speaker]["rate"],
                pitch=self.voice_config[speaker]["pitch"],
                output_path=output_path
            )
        except Exception as e:
            logger.error(f"MeloTTS synthesis failed: {e}")
            raise

    async def _synthesize_openai_tts(
        self,
        text: str,
        voice: str,
        output_path: str
    ) -> None:
        """OpenAI TTS로 음성 합성"""
        try:
            import openai

            client = openai.AsyncOpenAI()

            response = await client.audio.speech.create(
                model="tts-1-hd",
                voice=voice,  # alloy, echo, fable, onyx, nova, shimmer
                input=text,
                response_format="mp3"
            )

            # 파일로 저장
            with open(output_path, "wb") as f:
                async for chunk in response.iter_bytes():
                    f.write(chunk)

        except ImportError:
            logger.error("openai not installed. Run: pip install openai")
            raise
        except Exception as e:
            logger.error(f"OpenAI TTS synthesis failed: {e}")
            raise

    def _optimize_pronunciation(self, text: str) -> str:
        """발음 최적화 (전문용어 처리)"""
        for term, pronunciation in self.pronunciation_dict.items():
            text = text.replace(term, pronunciation)
        return text

    async def _get_audio_duration(self, audio_path: str) -> float:
        """오디오 길이 측정"""
        try:
            from pydub import AudioSegment as PydubSegment
            audio = PydubSegment.from_mp3(audio_path)
            return len(audio) / 1000.0  # milliseconds to seconds
        except ImportError:
            # pydub 없으면 추정
            return 30.0  # 기본값

    async def _merge_audio_segments(
        self,
        segments: List[AudioSegment],
        title: str
    ) -> Tuple[str, float]:
        """오디오 세그먼트 통합"""
        try:
            from pydub import AudioSegment as PydubSegment

            combined = PydubSegment.empty()

            for segment in segments:
                audio = PydubSegment.from_mp3(segment.audio_path)
                combined += audio

                # 턴 사이 짧은 무음 추가
                silence = PydubSegment.silent(duration=300)
                combined += silence

            # 최종 파일 저장
            safe_title = "".join(c for c in title if c.isalnum() or c in " _-")[:50]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"{safe_title}_{timestamp}.mp3"

            combined.export(str(output_path), format="mp3")

            return str(output_path), len(combined) / 1000.0

        except ImportError:
            logger.error("pydub not installed. Run: pip install pydub")
            # 첫 번째 세그먼트만 반환
            if segments:
                return segments[0].audio_path, segments[0].duration_seconds
            return "", 0.0

    async def generate_transcript(
        self,
        overview: AudioOverview,
        format: str = "srt"
    ) -> str:
        """자막 파일 생성"""
        if format == "srt":
            return self._generate_srt(overview.transcript)
        elif format == "vtt":
            return self._generate_vtt(overview.transcript)
        else:
            return self._generate_text_transcript(overview.transcript)

    def _generate_srt(self, dialogue: List[DialogueTurn]) -> str:
        """SRT 자막 형식 생성"""
        lines = []
        current_time = 0.0

        for i, turn in enumerate(dialogue, 1):
            # 예상 시간 (30초/턴)
            start_time = current_time
            end_time = current_time + 30.0

            start_str = self._format_srt_time(start_time)
            end_str = self._format_srt_time(end_time)

            speaker_label = "진행자" if turn.speaker == SpeakerRole.HOST_A else "질문자"

            lines.append(f"{i}")
            lines.append(f"{start_str} --> {end_str}")
            lines.append(f"[{speaker_label}] {turn.text}")
            lines.append("")

            current_time = end_time

        return "\n".join(lines)

    def _format_srt_time(self, seconds: float) -> str:
        """SRT 시간 형식"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _generate_vtt(self, dialogue: List[DialogueTurn]) -> str:
        """WebVTT 자막 형식 생성"""
        lines = ["WEBVTT", ""]
        srt_content = self._generate_srt(dialogue)
        # SRT를 VTT로 변환 (콤마→점)
        vtt_content = srt_content.replace(",", ".")
        lines.append(vtt_content)
        return "\n".join(lines)

    def _generate_text_transcript(self, dialogue: List[DialogueTurn]) -> str:
        """텍스트 대본 생성"""
        lines = []
        for turn in dialogue:
            speaker_label = "진행자" if turn.speaker == SpeakerRole.HOST_A else "질문자"
            lines.append(f"[{speaker_label}]: {turn.text}")
            lines.append("")
        return "\n".join(lines)
