import { Button, Card, Text } from "@mantine/core";
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
import { loadData } from "@utils/loadData";
import type { ICreateRecognize } from "@typings/recognize";
import { useNavigate } from "react-router";
import { notifications } from "@mantine/notifications";

export const Main = () => {
	const navigate = useNavigate();
	const [file, setFile] = useState<FileWithPath | null>(null);
	const [isLoading, setIsLoading] = useState(false);
	const hasFile = Boolean(file);

	const onDropHandler = (files: FileWithPath[]) => {
		setFile(files?.[0] || null);
	};

	const onClear = () => {
		setFile(null);
	};

	const sendToRecognation = async () => {
		if (!file) return;
		setIsLoading(true);

		try {
			const { task_id } = await loadData<ICreateRecognize>(
				"/recognize/create_task"
			);

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
		} catch (err) {
			console.error(`Ошибка при начале распознавания: ${err}`);
			notifications.show({
				color: "red",
				title: "Ошибка",
				message: "Ошибка при начале распознавания",
				position: "top-right",
				id: "error-when-starting-recognation",
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
					onClick={sendToRecognation}
					loading={isLoading}
					disabled={isLoading || !hasFile}
				>
					Отправить на распознавание
				</Button>
			</Card>
		</div>
	);
};
