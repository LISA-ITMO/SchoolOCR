import { Image, Modal } from "@mantine/core";

interface IProps {
	opened: boolean;
	onClose: () => void;
	imageUrl: string;
}

export const PreviewModal: React.FC<IProps> = ({
	opened,
	onClose,
	imageUrl,
}) => {
	return (
		<Modal
			opened={opened}
			onClose={onClose}
			centered
			shadow="xl"
			size="auto"
			transitionProps={{
				transition: "fade",
				duration: 100,
				timingFunction: "linear",
			}}
		>
			<Image src={imageUrl} w={700} />
		</Modal>
	);
};
