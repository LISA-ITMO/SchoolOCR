import { Button, Group, Progress, Text, Title } from "@mantine/core";
import styles from "./RecognizeHeader.module.css";

interface IProps {
	id?: string;
	isReady: boolean;
	completionPercent: number;
}

export const RecognizeHeader: React.FC<IProps> = ({
	id,
	isReady,
	completionPercent,
}) => {
	return (
		<div className={styles.root}>
			<div>
				<Title order={2}>Распознавание</Title>
				<Text c="dimmed">Задание: {id}</Text>
			</div>

			{isReady ? (
				<Button>Экспорт результатов</Button>
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
