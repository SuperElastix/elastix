/*
 * This file is part of the statismo library.
 *
 * Author: Marcel Luethi (marcel.luethi@unibas.ch)
 *
 * Copyright (c) 2011 University of Basel
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * Neither the name of the project's author nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS addINTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <boost/scoped_ptr.hpp>

#include "DataManager.h"
#include "PCAModelBuilder.h"
#include "StatisticalModel.h"
#include "StatismoIO.h"
#include "TrivialVectorialRepresenter.h"

typedef statismo::TrivialVectorialRepresenter RepresenterType;


/**
 * This basic test case, covers the model creation pipeline and tests whether a model can be successfully
 * saved to disk. If the test runs correctly, it merely means that statismo has been setup correclty and hdf5
 * works.
 *
 * Real unit tests that test the functionality of statismo are provided in the statismoTests directory (these tests
 * require VTK to be installed and the statismo python wrapping to be working).
 */
int main(int argc, char* argv[]) {

    typedef statismo::PCAModelBuilder<statismo::VectorType> ModelBuilderType;
    typedef statismo::StatisticalModel<statismo::VectorType> StatisticalModelType;
    typedef statismo::DataManager<statismo::VectorType> DataManagerType;


    try {
        const unsigned Dim = 3;
        boost::scoped_ptr<RepresenterType> representer(RepresenterType::Create(Dim));
        boost::scoped_ptr<DataManagerType> dataManager(DataManagerType::Create(representer.get()));

        // we create three simple datasets
        statismo::VectorType dataset1(Dim), dataset2(Dim), dataset3(Dim);
        dataset1 << 1,0,0;
        dataset2 << 0,2,0;
        dataset3 << 0,0,4;

        dataManager->AddDataset(dataset1, "dataset1");
        dataManager->AddDataset(dataset2, "dataset1");
        dataManager->AddDataset(dataset3, "dataset1");


        boost::scoped_ptr<ModelBuilderType> pcaModelBuilder(ModelBuilderType::Create());
        boost::scoped_ptr<StatisticalModelType> model(pcaModelBuilder->BuildNewModel(dataManager->GetData(), 0.01));

        // As we have added 3 linearly independent samples, we get 2 principal components.
        if (model->GetNumberOfPrincipalComponents() != 2) {
            return EXIT_FAILURE;
        }

        statismo::IO<statismo::VectorType>::SaveStatisticalModel(model.get(), "test.h5");

        RepresenterType* newRepresenter = RepresenterType::Create();
        boost::scoped_ptr<StatisticalModelType> loadedModel(
                statismo::IO<statismo::VectorType>::LoadStatisticalModel(newRepresenter, "test.h5"));
        if (model->GetNumberOfPrincipalComponents() != loadedModel->GetNumberOfPrincipalComponents()) {
            return EXIT_FAILURE;
        }


    } catch (statismo::StatisticalModelException& e) {
        std::cout << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;

}


