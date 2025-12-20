import { Box, Button, Card, Text } from "@mantine/core";
import { IconCheck } from "@tabler/icons-react";
import styles from "./UploadIsReady.module.css";

interface IUploadIsReady {
	fileName: string;
	fileSize: string;
	fileFormat: string;
	onClear: () => void;
}

export const UploadIsReady: React.FC<IUploadIsReady> = ({
	fileName,
	fileSize,
	fileFormat,
	onClear,
}) => {
	return (
		<Card shadow="sm" padding="lg" radius="md" withBorder>
			<div className={styles.wrapper}>
				<div className={styles.left}>
					<div>
						<IconCheck size={100} color="#18962fff" />
					</div>
					<div className={styles.fileInformation}>
						<Box w={300}>
							<Text fw={900} truncate="end">
								{fileName}
							</Text>
						</Box>

						<Text className={styles.text}>
							Загружен файл ({fileFormat.toUpperCase()}) — {fileSize} МБ
						</Text>
					</div>
				</div>
				<div>
					<Button
						variant="transparent"
						color="indigo"
						size="xl"
						onClick={onClear}
					>
						Удалить
					</Button>
				</div>
			</div>
		</Card>
	);
};
