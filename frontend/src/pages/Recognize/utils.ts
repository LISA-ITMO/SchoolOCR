import type { ScoresMap } from "@typings/recognize";

export const prepareToRow = (map: ScoresMap) => {
	const heading: string[] = [];
	const body: string[] = [];
	const accuracy: number[] = [];

	Object.entries(map).forEach(([headCellValue, [bodyCellValue, accValue]]) => {
		heading.push(headCellValue);
		body.push(bodyCellValue);
		accuracy.push(accValue);
	});

	return { heading, body, accuracy };
};

export const sumScores = (map: ScoresMap) =>
	Object.values(map).reduce((acc, [v]) => acc + (Number(v) || 0), 0);
