import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors",
  {
    variants: {
      variant: {
        default: "border-amber-500/30 bg-amber-500/15 text-amber-300",
        emerald: "border-emerald-500/30 bg-emerald-500/15 text-emerald-300",
        rose: "border-rose-500/30 bg-rose-500/15 text-rose-300",
        sky: "border-sky-500/30 bg-sky-500/15 text-sky-300",
        outline: "border-slate-700 bg-slate-800/60 text-slate-300",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />;
}

export { Badge, badgeVariants };
