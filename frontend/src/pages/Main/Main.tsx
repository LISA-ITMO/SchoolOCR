import { Button, Card, Text, SegmentedControl, TextInput, JsonInput, Collapse } from "@mantine/core";
import styles from "./Main.module.css";
import { DropZone } from "@components/DropZone/DropZone";
import { useState } from "react";
import type { FileWithPath } from "@mantine/dropzone";
import { UploadIsReady } from "@components/UploadIsReady/UploadIsReady";
import {
	ACCEPT_MIME_TYPES,
	KEYMAP_MIME_TYPES_TO_FORMAT,
} from "@constants/index";
import { bitToMb } from "@utils/bitToMb";
import { loadData, SERVICE_API_KEY_STORAGE_KEY } from "@utils/loadData";
import type { ICreateRecognize } from "@typings/recognize";
import { useNavigate } from "react-router";
import { notifications } from "@mantine/notifications";

type Mode = "ocr" | "llm";

export const Main = () => {
	const navigate = useNavigate();
	const [file, setFile] = useState<FileWithPath | null>(null);
	const [isLoading, setIsLoading] = useState(false);
	const [mode, setMode] = useState<Mode>("ocr");
	const [serviceApiKey, setServiceApiKey] = useState(
		() => localStorage.getItem(SERVICE_API_KEY_STORAGE_KEY) ?? ""
	);
	const [ollamaApiKey, setOllamaApiKey] = useState("");
	const [llmResult, setLlmResult] = useState<string | null>(null);
	const hasFile = Boolean(file);

	const onDropHandler = (files: FileWithPath[]) => {
		setFile(files?.[0] || null);
		setLlmResult(null);
	};

	const onClear = () => {
		setFile(null);
		setLlmResult(null);
	};

	const sendOcr = async () => {
		if (!file) return;

		const { task_id } = await loadData<ICreateRecognize>("/recognize/create_task");

		const formData = new FormData();
		formData.append("file", file);

		await loadData(`/recognize/${task_id}/start`, {
			method: "POST",
			body: formData,
		});

		notifications.show({
			title: "Успех",
			message: "Распознавание вашего документа успешно началось!",
			position: "top-right",
			id: "starting-recognation",
		});

		navigate(`/recognize/${task_id}`);
	};

	const sendLlm = async () => {
		if (!file) return;

		const formData = new FormData();
		formData.append("file", file);
		if (ollamaApiKey) {
			formData.append("api_key", ollamaApiKey);
		}

		const result = await loadData<Record<string, unknown>>("/llm/recognize", {
			method: "POST",
			body: formData,
		});

		setLlmResult(JSON.stringify(result, null, 2));

		notifications.show({
			title: "Успех",
			message: "LLM распознавание завершено",
			position: "top-right",
			id: "llm-success",
		});
	};

	const onSubmit = async () => {
		if (!file) return;
		setIsLoading(true);
		try {
			if (mode === "ocr") {
				await sendOcr();
			} else {
				await sendLlm();
			}
		} catch (err) {
			console.error(`Ошибка при распознавании: ${err}`);
			notifications.show({
				color: "red",
				title: "Ошибка",
				message: "Ошибка при распознавании",
				position: "top-right",
				id: "error-when-recognating",
			});
		} finally {
			setIsLoading(false);
		}
	};

	return (
		<div className={styles.main}>
			<Card
				shadow="sm"
				padding="lg"
				radius="md"
				withBorder
				className={styles.card}
			>
				<Text
					size="xl"
					variant="gradient"
					fw={900}
					gradient={{ from: "violet", to: "blue", deg: 145 }}
					className={styles.mainText}
				>
					Сервис для распознавания информации с бланков ВПР
				</Text>

				<SegmentedControl
					value={mode}
					onChange={(v) => { setMode(v as Mode); setLlmResult(null); }}
					data={[
						{ label: "OCR-модель", value: "ocr" },
						{ label: "LLM-модель", value: "llm" },
					]}
					mb="md"
					fullWidth
				/>

				<TextInput
					label="API-ключ сервиса"
					placeholder="X-API-Key"
					value={serviceApiKey}
					onChange={(e) => {
						const val = e.currentTarget.value;
						setServiceApiKey(val);
						localStorage.setItem(SERVICE_API_KEY_STORAGE_KEY, val);
					}}
					mb="sm"
				/>

				<Collapse in={mode === "llm"}>
					<TextInput
						label="API-ключ Ollama"
						placeholder="Ключ для Ollama"
						value={ollamaApiKey}
						onChange={(e) => setOllamaApiKey(e.currentTarget.value)}
						mb="sm"
					/>
				</Collapse>

				{!hasFile ? (
					<DropZone
						textMain="Переместите изображение сюда или кликните для выбора файла"
						textDescription="Поддерживаемые форматы: pdf, png, jpeg"
						accept={ACCEPT_MIME_TYPES}
						onDrop={onDropHandler}
						onReject={(m) => console.log(m)}
					/>
				) : (
					<UploadIsReady
						fileName={file?.name as string}
						fileSize={bitToMb(file?.size as number)}
						fileFormat={KEYMAP_MIME_TYPES_TO_FORMAT[file?.type as string]}
						onClear={onClear}
					/>
				)}

				<Button
					variant="light"
					color="indigo"
					size="xl"
					className={styles.btn}
					onClick={onSubmit}
					loading={isLoading}
					disabled={isLoading || !hasFile}
				>
					{mode === "ocr" ? "Отправить на распознавание" : "Распознать через LLM"}
				</Button>

				{llmResult && (
					<JsonInput
						mt="md"
						label="Результат LLM"
						value={llmResult}
						readOnly
						autosize
						minRows={6}
						maxRows={20}
						formatOnBlur
					/>
				)}
			</Card>
		</div>
	);
};
