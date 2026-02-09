
import { GoogleGenAI, Type, GenerateContentResponse, Modality } from "@google/genai";
import { ResponseMode } from "../types";

// Helper para obter a instância do AI com a chave de ambiente
const getAI = () => {
  const apiKey = process.env.API_KEY;
  if (!apiKey) {
    throw new Error("API_KEY não encontrada. Verifique as variáveis de ambiente no Netlify.");
  }
  return new GoogleGenAI({ apiKey });
};

const cleanOutput = (text: string): string => {
  if (!text) return "";
  return text
    .replace(/[\$\*\_\#]/g, '')
    .replace(/\\\[/g, '')
    .replace(/\\\]/g, '')
    .replace(/\\\(/g, '')
    .replace(/\\\)/g, '')
    .trim();
};

export const solveFromImage = async (
  base64Image: string, 
  thinking: boolean = true, 
  mode: ResponseMode = ResponseMode.EXPLAINED
): Promise<string> => {
  const ai = getAI();
  const model = thinking ? 'gemini-3-pro-preview' : 'gemini-3-flash-preview';
  
  const simplePrompt = `ATUE COMO UM RESOLVEDOR DIRETO (MODO SIMPLES). Forneça APENAS o resultado. PROIBIDO símbolos como '$', '*', '_' ou '#'.`;
  const explainedPrompt = `ATUE COMO UM RESOLVEDOR DIRETO (MODO EXPLICAÇÃO). Forneça o resultado e uma explicação curta de 2 frases. PROIBIDO símbolos como '$', '*', '_' ou '#'.`;

  const prompt = mode === ResponseMode.SIMPLE ? simplePrompt : explainedPrompt;

  const response = await ai.models.generateContent({
    model: model,
    contents: {
      parts: [
        { inlineData: { mimeType: "image/jpeg", data: base64Image } },
        { text: prompt }
      ]
    },
    config: {
      thinkingConfig: thinking ? { thinkingBudget: 16384 } : undefined,
      temperature: 0.1,
    }
  });

  return cleanOutput(response.text || "Não foi possível processar.");
};

export const chatWithProfessor = async (
  message: string, 
  mode: ResponseMode = ResponseMode.EXPLAINED
): Promise<string> => {
  const ai = getAI();
  const instruction = mode === ResponseMode.SIMPLE 
    ? 'Resolva diretamente sem explicações ou símbolos.' 
    : 'Resolva e explique brevemente sem símbolos.';

  const chat = ai.chats.create({
    model: 'gemini-3-pro-preview',
    config: {
      systemInstruction: instruction,
      thinkingConfig: { thinkingBudget: 8192 }
    }
  });

  const response = await chat.sendMessage({ message });
  return cleanOutput(response.text || "Erro no processamento.");
};

export const generateSpeech = async (text: string): Promise<Uint8Array | null> => {
  try {
    const ai = getAI();
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash-preview-tts",
      contents: [{ parts: [{ text: text }] }],
      config: {
        responseModalities: [Modality.AUDIO],
        speechConfig: {
          voiceConfig: {
            prebuiltVoiceConfig: { voiceName: 'Kore' },
          },
        },
      },
    });

    const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
    if (base64Audio) {
      const binaryString = atob(base64Audio);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      return bytes;
    }
    return null;
  } catch (error) {
    console.error("Erro ao gerar áudio:", error);
    return null;
  }
};
