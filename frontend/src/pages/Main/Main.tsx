import { Card, Text } from "@mantine/core";
import styles from "./Main.module.css";

export const Main = () => {
	return (
		<div className={styles.main}>
			<Card
				shadow="sm"
				padding="lg"
				radius="md"
				withBorder
				className={styles.card}
			>
				<Text size="xl">VPR lists recognation</Text>
			</Card>
		</div>
	);
};
