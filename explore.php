<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Loggers\Screen;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Extractors\NDJSON;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Transformers\PrincipalComponentAnalysis;
use Rubix\ML\Transformers\LinearDiscriminantAnalysis;
use Rubix\ML\Transformers\TruncatedSVD;
use Rubix\ML\Transformers\TSNE;

ini_set('memory_limit', '-1');

$logger = new Screen();

$logger->info('Loading data into memory');

$dataset = Labeled::fromIterator(new NDJSON('dataset.ndjson'));

$embedder = new TSNE(2, 100.0, 10, 6.0, 1000, 1e-40);

$embedder->setLogger($logger);

$dataset ->apply($embedder)
    ->exportTo(new CSV('tsne.csv'));
