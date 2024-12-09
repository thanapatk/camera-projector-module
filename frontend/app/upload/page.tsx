"use client";
import { motion } from "framer-motion";
import { useState } from "react";
import Image from "next/image";
import { useRouter } from "next/navigation";
import { LoadingTransition } from "@/components/LoadingTransition";

export default function ImageUploadPage() {
    const [selectedImage, setSelectedImage] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const router = useRouter();

    const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file && file.type === "image/png") {
            const reader = new FileReader();
            reader.onloadend = () => {
                setSelectedImage(reader.result as string);
            };
            reader.readAsDataURL(file);
        }
    };

    const handleUploadImage = async () => {
        if (selectedImage) {
            setIsLoading(true);

            try {
                const response = await fetch(
                    "http://thanapatk.local:8000/upload_image",
                    {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json",
                        },
                        body: JSON.stringify({ image: selectedImage }),
                    },
                );

                if (!response.ok) throw Error("Network response was no ok");

                window.localStorage.setItem("uploadedImage", selectedImage);
                router.push("/project");
            } catch (error) {
                console.error("Error starting projector:", error);
                setIsLoading(false);
            }
        }
    };

    // Animation Variants
    const fadeInUpVariants = {
        hidden: { opacity: 0, y: 20 },
        visible: {
            opacity: 1,
            y: 0,
            transition: { duration: 0.8, ease: "easeOut", staggerChildren: 0.3 },
        },
    };

    //const fadeInVariants = {
    //    hidden: { opacity: 0 },
    //    visible: { opacity: 1, transition: { duration: 0.8, ease: "easeOut" } },
    //};

    return (
        <motion.div
            className="min-h-screen bg-black p-8 flex flex-col md:flex-row items-center justify-center gap-[64px]"
            initial="hidden"
            animate="visible"
            variants={fadeInUpVariants}
        >
            {/* Left side - Text */}
            <motion.div
                className="text-white space-y-2 pr-6"
                variants={fadeInUpVariants}
            >
                <h1 className="text-6xl pb-2">
                    Upload
                    <br />
                    Your Image
                </h1>
                <p className="text-gray-400 text-[18px]">
                    images must be in .png and has a 1:1 ratio
                </p>
            </motion.div>

            {/* Right side - Upload area */}
            <motion.div
                className="w-full max-w-md space-y-4"
                variants={fadeInUpVariants}
            >
                <div className="relative aspect-square rounded-3xl border-2 border-dashed bg-gray-700 flex items-center justify-center">
                    {selectedImage ? (
                        <Image
                            src={selectedImage}
                            alt="Uploaded preview"
                            fill
                            className="object-cover rounded-3xl"
                        />
                    ) : (
                        <span className="text-white text-6xl">+</span>
                    )}
                    <input
                        type="file"
                        accept=".png"
                        onChange={handleImageChange}
                        className="absolute inset-0 opacity-0 cursor-pointer"
                    />
                </div>

                {/* Button - Fade In Only */}
                <button
                    className="w-full py-4 bg-blue-600 text-white rounded-xl font-semibold hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    disabled={!selectedImage}
                    onClick={handleUploadImage}
                >
                    CONTINUE
                </button>
            </motion.div>

            <LoadingTransition isLoading={isLoading} text="Uploading" />
        </motion.div>
    );
}
