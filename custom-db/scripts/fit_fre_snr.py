#!/usr/bin/env python

import luigi
import maprfanalysis.bbo as bboutils
from maprfanalysis.bbo.tasks import ComputeEffectiveFreAndSNR


excluded = bboutils.models.query('output=="ENERGY" or invlink == "EXP"').index
fits = bboutils.fits
fits = bboutils.fits.assign(length=fits.index2-fits.index1)\
	.query('length==7200 & sampler=="fancy" & model_id not in @excluded')\
	.drop("length", axis=1)


def main():
	luigi.build([mktask(fit_id) for fit_id in fits.index])


def mktask(fit_id):
	obj = bboutils.AnalysisTask(fit_id)
	obj.postprocess = [ComputeEffectiveFreAndSNR]
	return obj


if __name__ == '__main__':
	luigi.build([mktask(fit_id) for fit_id in fits.index])
