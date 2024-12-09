import { motion, AnimatePresence } from "framer-motion";
import { useEffect, useState } from "react";

interface LoadingTransitionProps {
    isLoading: boolean;
    text: string;
}

export const LoadingTransition = ({
    isLoading,
    text,
}: LoadingTransitionProps) => {
    const [dotCount, setDotCount] = useState(0);

    // Cycle through dots for the loading text
    useEffect(() => {
        if (isLoading) {
            const interval = setInterval(() => {
                setDotCount((prev) => (prev + 1) % 4); // Cycle between 0, 1, 2, 3
            }, 500); // Change dots every 500ms

            return () => clearInterval(interval); // Cleanup interval on unmount
        }
    }, [isLoading]);

    const loadingText = `${text}${".".repeat(dotCount)}`;

    return (
        <AnimatePresence>
            {isLoading && (
                <motion.div
                    className="absolute inset-0 flex items-center justify-center backdrop-blur-2xl z-50"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.8 }}
                >
                    <p className="text-[24px] tracking-wide">{loadingText}</p>
                </motion.div>
            )}
        </AnimatePresence>
    );
};
