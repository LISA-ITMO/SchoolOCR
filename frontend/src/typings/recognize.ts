export type TaskScore = [value: string, confidence: number];

export type ScoresMap = Record<string, TaskScore>;

export type ScoreDetails = Record<string, number>;

export type ScoresDetailsMap = Record<string, ScoreDetails>;

export interface IRecognizeItem {
	id: string;
	grade: number | null;
	errors: string[] | null;
	warnings: string[] | null;

	subject: string | null;
	variant: string | null;

	participant_code: string | null;

	image_url: string;
	total_score: number;

	scores: ScoresMap;
	scores_details: ScoresDetailsMap;
}

export interface IRecognizeResponse {
	items: IRecognizeItem[];
	is_ready: boolean;
	completion_percent: number;
}

export interface IRecognizeGetIsReady {
	is_ready: true;
	completion_percent: number;
}

export interface IUploadRecognize {
	file: FormData;
}

export interface ICreateRecognize {
	task_id: string;
}
