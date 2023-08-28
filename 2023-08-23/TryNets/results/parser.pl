#!/usr/bin/perl

my @lines = <>;

print join("\t", "period", "acc0", "acc1", "acc2", "accuracy", "time0", "time1", "time2", "time", "rate0", "rate1", "rate2", "accpick0", "accpick1", "accpick2"), "\n";

while (@lines) {
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
    ($accexit0, $accexit1, $accexit2) = $line =~ /([\d\.]+)% \| ([\d\.]+)% \| ([\d\.]+)%/;
    $line = shift @lines;
    ($accuracy) = $line =~ /: ([\d\.]+)/;
    $line = shift @lines;
    ($meantime) = $line =~ /: ([\d\.]+)/;
    $line = join("\t", $period, $acc0, $acc1, $acc2, $accuracy, $time0, $time1, $time2, $meantime, $rate0, $rate1, $rate2, $accexit0, $accexit1, $accexit2);
    $line =~ s/\./,/g;
    print "$line\n";
}

#2016-01: Accuracy: 90.71 | 91.36 | 93.86
#Thresholds: 0.7 | 0.7 | 0
#Mean times per exit: 0.1547 ms | 0.5489 ms | 1.176 ms
#Accuracies per exit (for all dataset): 90.7136%  | 91.3561%  | 93.8559%
#Rate of exit chosen: 89.32% | 4.52% | 6.16%
#Accuracy per exit (when chosen): 94.56% | 81.83% | 87.20%
#Overall Accuracy: 93.53%
#Mean time: 0.2354 ms


