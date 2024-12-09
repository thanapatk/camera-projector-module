"use client";

import { useRouter } from "next/navigation";
import { Button } from "@/components/Button";
import { LoadingTransition } from "@/components/LoadingTransition";
import { useEffect, useState } from "react";
import { motion } from "framer-motion";

export default function Home() {
    const router = useRouter();
    const [error, setError] = useState<string>();
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        async function fetchData() {
            try {
                const response = await fetch(
                    "http://thanapatk.local:8000/projector_started",
                );
                if (!response.ok) {
                    throw new Error("Network response was not ok");
                }

                const data = await response.json();
                if (data.started) {
                    router.replace("/upload");
                } else {
                    setIsLoading(false);
                }
            } catch (error) {
                console.error("Error fetching data:", error);
                setError("Something went wrong! Please reload the page.");
                setIsLoading(false); // Prevent infinite loading on error
            }
        }

        fetchData();
    }, [router]);

    const handleProjectClick = async ({ calibrate }: { calibrate: boolean }) => {
        setIsLoading(true); // Show loading screen

        try {
            const response = await fetch(
                "http://thanapatk.local:8000/start_projector",
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({ calibrate }),
                },
            );

            if (!response.ok) throw Error("Network response was no ok");

            router.replace("/upload");
        } catch (error) {
            console.error("Error starting projector:", error);
            setIsLoading(false);
        }
    };

    // Animation variants
    const containerVariants = {
        hidden: { opacity: 0, y: 20 },
        visible: {
            opacity: 1,
            y: 0,
            transition: { duration: 0.8, ease: "easeOut", staggerChildren: 0.3 },
        },
    };

    const fadeInVariants = {
        hidden: { opacity: 0 },
        visible: { opacity: 1, transition: { duration: 0.8, ease: "easeOut" } },
    };

    if (error)
        return (
            <main className="grid place-items-center h-screen">
                <h1 className="text-2xl text-center">
                    Something went wrong...
                    <br />
                    Please reload the page
                </h1>
            </main>
        );

    return (
        <motion.main
            className="flex min-h-screen flex-col items-center justify-center bg-[url('/Home.png')] p-8"
            initial="hidden"
            animate="visible"
            variants={containerVariants}
        >
            {/* Animated Content */}
            <motion.div className="text-center" variants={containerVariants}>
                {/* Image */}
                <motion.img
                    src="/bi_projector.png"
                    className="mx-auto pb-2"
                    alt="BI Projector"
                    variants={fadeInVariants}
                />

                {/* Heading */}
                <motion.h1
                    className="text-7xl text-white mb-8"
                    variants={containerVariants}
                >
                    Get Started
                </motion.h1>

                {/* Buttons Container */}
                <div className="space-x-4 flex flex-row">
                    {/* Button 1 - Fade In Only */}
                    <motion.div variants={fadeInVariants}>
                        <Button onClick={() => handleProjectClick({ calibrate: false })}>
                            PROJECT WITHOUT CALIBRATION
                        </Button>
                    </motion.div>

                    {/* Button 2 - Fade In Only */}
                    <motion.div variants={fadeInVariants}>
                        <Button
                            variant="secondary"
                            onClick={() => handleProjectClick({ calibrate: true })}
                        >
                            CALIBRATE
                        </Button>
                    </motion.div>
                </div>
            </motion.div>

            {/* Loading Overlay */}
            <LoadingTransition isLoading={isLoading} text="Starting Projector" />
        </motion.main>
    );
}
