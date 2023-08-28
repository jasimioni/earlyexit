#!/usr/bin/perl

my @lines = <>;

print join("\t", "period", "acc0", "acc1", "acc2", "accuracy", "time0", "time1", "time2", "time", "rate0", "rate1", "rate2", "accpick0", "accpick1", "accpick2"), "\n";

while (@lines) {
    shift @lines;
    shift @lines;
    $line = shift @lines;
    chomp $line;

    ($period, $acc0, $acc1, $acc2) = $line =~ /^(.*): Accuracy: ([\d+\.]+) \| ([\d+\.]+) \| ([\d+\.]+)/;
    shift @lines;
    $line = shift @lines;
    ($time0, $time1, $time2) = $line =~ /([\d\.]+) ms \| ([\d\.]+) ms \| ([\d\.]+) ms/;
    shift @lines;
    $line = shift @lines;
    ($rate0, $rate1, $rate2) = $line =~ /([\d\.]+)% \| ([\d\.]+)% \| ([\d\.]+)%/;
    $line = shift @lines;
    ($accexit0, $accexit1, $accexit2) = $line =~ /([\d\.]+) % \| ([\d\.]+) % \| ([\d\.]+) %/;
    $line = shift @lines;
    ($accuracy) = $line =~ /: ([\d\.]+)/;
    $line = shift @lines;
    ($meantime) = $line =~ /: ([\d\.]+)/;
    $line = join("\t", $period, $acc0, $acc1, $acc2, $accuracy, $time0, $time1, $time2, $meantime, $rate0, $rate1, $rate2, $accexit0, $accexit1, $accexit2);
    $line =~ s/\./,/g;
    print "$line\n";
}

# Getting files from ../../datasets/scaled/MOORE/2016/01
# 2016-01
# 2016-01: Accuracy: 71.78 | 81.00 | 90.47
# Thresholds: 0.7 | 0.7 | 0
# Mean times per exit: 0.1299 ms | 0.282 ms | 0.5019 ms
# Accuracies per exit (for all dataset): 71.781%  | 80.9972%  | 90.4733%
# Rate of exit chosen: 34.40% | 53.86% | 11.74%
# Accuracy per exit (when chosen): 90.00 % | 90.23 % | 91.99 %
# Overall Accuracy: 90.36%
# Mean time: 0.2555 ms
# =cut
