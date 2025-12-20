import { Loader, Text } from "@mantine/core";
import type { IRecognizeItem, IRecognizeResponse } from "@typings/recognize";
import { loadData } from "@utils/loadData";
import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router";
import { useDisclosure } from "@mantine/hooks";

import styles from "./Recognize.module.pcss";
import { prepareToRow, sumScores } from "./utils";
import { PreviewModal } from "./components/PreviewModal/PreviewModal";
import { RecognizeHeader } from "./components/RecognizeHeader/RecognizeHeader";
import { ParticipantCard } from "./components/ParticipantCard/ParticipantCard";

export const Recognize = () => {
	const { id } = useParams();
	const navigate = useNavigate();

	const [opened, { open, close }] = useDisclosure(false);
	const [selectedImgUrl, setSelectedImageUrl] = useState("");

	const [result, setResults] = useState<IRecognizeItem[] | null>(null);
	const [completionPercent, setCompletionPercent] = useState(0);
	const [isInitialLoading, setIsInitialLoading] = useState(false);

	const isReady = completionPercent === 100;

	const rowsData = useMemo(
		() => (result || []).map((item) => prepareToRow(item.scores)),
		[result]
	);

	const openPreviewModal = useCallback(
		(src: string) => {
			setSelectedImageUrl(src);
			open();
		},
		[open]
	);

	const startPullingResults = useCallback(() => {
		const intervalId = setInterval(async () => {
			const response = await loadData<IRecognizeResponse>(`/recognize/${id}`);
			setResults(response.items);
			setCompletionPercent(response.completion_percent);

			if (response.is_ready) clearInterval(intervalId);
		}, 1000);

		return () => clearInterval(intervalId);
	}, [id]);

	const initialLoad = useCallback(async () => {
		setIsInitialLoading(true);
		try {
			const response = await loadData<IRecognizeResponse>(`/recognize/${id}`);
			setResults(response.items);
			setCompletionPercent(response.completion_percent);

			if (!response.is_ready) startPullingResults();
		} catch (err) {
			if ((err as { status: number }).status === 404) navigate("/notFound");
		} finally {
			setIsInitialLoading(false);
		}
	}, [id, navigate, startPullingResults]);

	useEffect(() => {
		initialLoad();
	}, [initialLoad]);

	if (isInitialLoading) {
		return (
			<div className={styles.pageCenter}>
				<Loader color="blue" />
				<Text c="dimmed" mt="sm">
					Загружаем результаты…
				</Text>
			</div>
		);
	}

	return (
		<div className={styles.page}>
			<PreviewModal opened={opened} onClose={close} imageUrl={selectedImgUrl} />

			<RecognizeHeader
				id={id}
				isReady={isReady}
				completionPercent={completionPercent}
			/>

			<div className={styles.cards}>
				{result?.map((item, idx) => (
					<ParticipantCard
						key={`${item.participant_code}-${idx}`}
						item={item}
						row={rowsData[idx]}
						total={sumScores(item.scores)}
						onPreviewClick={openPreviewModal}
					/>
				))}
			</div>
		</div>
	);
};
