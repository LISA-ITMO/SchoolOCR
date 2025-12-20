import { Card, Image, Text } from "@mantine/core";
import styles from "./PreviewCard.module.css";

interface IProps {
	imageUrl: string;
	onClick: () => void;
}

export const PreviewCard: React.FC<IProps> = ({ imageUrl, onClick }) => {
	return (
		<>
			<Text fw={700} mb="xs">
				Превью бланка
			</Text>

			<Card
				withBorder
				radius="md"
				className={styles.previewCard}
				onClick={onClick}
			>
				<Image
					src={imageUrl}
					alt="Бланк"
					radius="md"
					fit="contain"
					className={styles.previewImg}
				/>
				<Text size="xs" c="dimmed" mt="xs">
					*Нажмите на изображение для открытия
				</Text>
			</Card>
		</>
	);
};
