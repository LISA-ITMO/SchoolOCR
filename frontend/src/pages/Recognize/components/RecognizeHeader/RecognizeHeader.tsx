import { Button, Group, Progress, Text, Title } from "@mantine/core";
import styles from "./RecognizeHeader.module.css";
import type { IRecognizeItem } from "@typings/recognize";
import { useCallback, useMemo } from "react";
import { downloadJson } from "@utils/downloadJson";

interface IProps {
	id?: string;
	isReady: boolean;
	completionPercent: number;
	items: IRecognizeItem[];
	deletedIds: string[];
}

export const RecognizeHeader: React.FC<IProps> = ({
	id,
	isReady,
	completionPercent,
	items,
	deletedIds,
}) => {
	const preparedResults = useMemo(
		() =>
			items
				.filter((item) => !deletedIds.includes(item.id))
				.map((item) => {
					const scoresFormated = Object.entries(item.scores).reduce(
						(acc, entry) => {
							const [headCell, [recognizedClass]] = entry;

							return { ...acc, [headCell]: recognizedClass };
						},
						{}
					);

					return {
						image_url: item.image_url,
						participant_code: item.participant_code,
						scores: scoresFormated,
						subject: item.subject,
						grade: item.grade,
						variant: item.variant,
					};
				}),
		[deletedIds, items]
	);
	const handleDownload = useCallback(() => {
		downloadJson(preparedResults, "result.json");
	}, [preparedResults]);

	return (
		<div className={styles.root}>
			<div>
				<Title order={2}>Распознавание</Title>
				<Text c="dimmed">Задание: {id}</Text>
			</div>

			{isReady ? (
				<Button onClick={handleDownload}>Экспорт результатов</Button>
			) : (
				<div className={styles.progressBlock}>
					<Group justify="space-between" gap="xs" wrap="nowrap">
						<Text size="sm" c="dimmed">
							Прогресс
						</Text>
						<Text size="sm" fw={600}>
							{completionPercent}%
						</Text>
					</Group>
					<Progress value={completionPercent} radius="xl" animated />
				</div>
			)}
		</div>
	);
};
