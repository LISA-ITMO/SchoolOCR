import { Button, Card, Group, Stack, Text } from "@mantine/core";
import type { IRecognizeItem } from "@typings/recognize";
import styles from "./ParticipantCard.module.css";
import { ScoresTable } from "../ScoresTable/ScoresTable";
import { PreviewCard } from "../PreviewCard/PreviewCard";

type RowData = {
	heading: string[];
	body: (string | number)[];
	accuracy: number[];
};

interface IProps {
	item: IRecognizeItem;
	row: RowData;
	total: number;
	onPreviewClick: (url: string) => void;
	deleteCard: (id: string) => void;
	isDeleted: boolean;
}

export const ParticipantCard: React.FC<IProps> = ({
	item,
	row,
	total,
	onPreviewClick,
	deleteCard,
	isDeleted,
}) => {
	if (isDeleted) return null;

	return (
		<Card radius="lg" withBorder className={styles.card}>
			<Card.Section className={styles.cardHeader}>
				<Group justify="space-between" align="flex-end" wrap="nowrap">
					<Text fw={800} size="xl">
						Сумма баллов: {total}
					</Text>
					<Button color="red" onClick={() => deleteCard(item.id)}>
						Удалить из выгрузки
					</Button>
				</Group>
			</Card.Section>

			<div className={styles.content}>
				<div className={styles.left}>
					<Stack gap="sm">
						<div className={styles.metaGrid}>
							<div className={styles.metaItem}>
								<Text size="xs" c="dimmed">
									Код участника
								</Text>
								<Text fw={600}>{item.participant_code}</Text>
							</div>

							<div className={styles.metaItem}>
								<Text size="xs" c="dimmed">
									Предмет
								</Text>
								<Text fw={600}>{item.subject}</Text>
							</div>

							<div className={styles.metaItem}>
								<Text size="xs" c="dimmed">
									Вариант
								</Text>
								<Text fw={600}>{item.variant}</Text>
							</div>
						</div>

						<ScoresTable row={row} />
					</Stack>
				</div>

				<div className={styles.right}>
					<PreviewCard
						imageUrl={item.image_url}
						onClick={() => onPreviewClick(item.image_url)}
					/>
				</div>
			</div>
		</Card>
	);
};
