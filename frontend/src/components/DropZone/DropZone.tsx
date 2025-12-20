import { Group, Text } from "@mantine/core";
import {
	Dropzone as DropZoneBase,
	type FileRejection,
	type FileWithPath,
} from "@mantine/dropzone";
import { IconFile, IconX } from "@tabler/icons-react";

interface IDropZone {
	onDrop?: (files: FileWithPath[]) => void;
	onReject?: (files: FileRejection[]) => void;
	maxSize?: number;
	accept: string[];
	textMain: string;
	textDescription: string;
}

export const DropZone: React.FC<IDropZone> = ({
	onDrop,
	onReject,
	maxSize,
	accept,
	textMain,
	textDescription,
}) => {
	return (
		<DropZoneBase
			onDrop={onDrop || (() => {})}
			onReject={onReject || (() => {})}
			maxSize={maxSize}
			accept={accept}
		>
			<Group
				justify="center"
				gap="xl"
				mih={220}
				style={{ pointerEvents: "none" }}
			>
				<DropZoneBase.Accept>
					<IconFile
						size={52}
						color="var(--mantine-color-blue-6)"
						stroke={1.5}
					/>
				</DropZoneBase.Accept>
				<DropZoneBase.Reject>
					<IconX size={52} color="var(--mantine-color-red-6)" stroke={1.5} />
				</DropZoneBase.Reject>
				<DropZoneBase.Idle>
					<IconFile
						size={52}
						color="var(--mantine-color-dimmed)"
						stroke={1.5}
					/>
				</DropZoneBase.Idle>

				<div>
					<Text size="xl" inline>
						{textMain}
					</Text>
					<Text size="sm" c="dimmed" inline mt={7}>
						{textDescription}
					</Text>
				</div>
			</Group>
		</DropZoneBase>
	);
};
