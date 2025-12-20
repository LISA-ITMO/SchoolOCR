import { ScrollArea, Table, Text, Tooltip } from "@mantine/core";
import { IconAlertCircleFilled } from "@tabler/icons-react";
import styles from "./ScoresTable.module.css";

interface IProps {
	row: {
		heading: string[];
		body: (string | number)[];
		accuracy: number[];
	};
}

export const ScoresTable: React.FC<IProps> = ({ row }) => {
	return (
		<div className={styles.tableWrap}>
			<Text fw={700} mb="xs">
				Результаты по заданиям
			</Text>

			<ScrollArea type="auto" offsetScrollbars>
				<Table
					withTableBorder
					withColumnBorders
					highlightOnHover
					verticalSpacing="sm"
					horizontalSpacing="md"
					className={styles.table}
				>
					<Table.Thead>
						<Table.Tr>
							<Table.Th className={styles.firstCol}>Задание</Table.Th>
							{row.heading.map((cell) => (
								<Table.Th key={cell}>{cell}</Table.Th>
							))}
						</Table.Tr>
					</Table.Thead>

					<Table.Tbody>
						<Table.Tr>
							<Table.Td className={styles.firstCol}>
								<Text fw={600}>Баллы</Text>
							</Table.Td>

							{row.body.map((cell, i) => {
								const acc = row.accuracy?.[i] ?? 1;
								const isWarn = acc < 0.7;

								return (
									<Table.Td key={i}>
										<div className={styles.cell}>
											<Text fw={600}>{cell}</Text>

											{isWarn && (
												<Tooltip
													label={`Внимание! Вероятность распознавания ${Math.round(
														acc * 100
													)}%`}
												>
													<IconAlertCircleFilled color="red" />
												</Tooltip>
											)}
										</div>
									</Table.Td>
								);
							})}
						</Table.Tr>
					</Table.Tbody>
				</Table>
			</ScrollArea>
		</div>
	);
};
