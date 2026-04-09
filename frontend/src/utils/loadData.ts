export const SERVICE_API_KEY_STORAGE_KEY = "service_api_key";

export const loadData = async <T>(path: string, params?: RequestInit) => {
	try {
		const baseUrl = "http://localhost:8000";
		const url = `${baseUrl}${path}`;

		const storedKey = localStorage.getItem(SERVICE_API_KEY_STORAGE_KEY);
		const authHeaders: Record<string, string> = storedKey
			? { "X-API-Key": storedKey }
			: {};

		const mergedParams: RequestInit = {
			...params,
			headers: { ...authHeaders, ...(params?.headers as Record<string, string> | undefined) },
		};

		const request = await fetch(url, mergedParams);
		if (!request.ok) {
			throw request;
		}

		const json = await request.json();

		return json as Promise<T>;
	} catch (err) {
		console.error(`error when requesting ${path}: ${err}`);
		throw err;
	}
};
