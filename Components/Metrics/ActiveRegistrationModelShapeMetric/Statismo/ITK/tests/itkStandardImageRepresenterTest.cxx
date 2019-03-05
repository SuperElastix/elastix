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
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "genericRepresenterTest.hxx"
#include "itkStandardImageRepresenter.h"


typedef itk::Image< float,2 > ScalarImageType;


ScalarImageType::Pointer loadScalarImage(const std::string& filename) {
    itk::ImageFileReader<ScalarImageType>::Pointer reader = itk::ImageFileReader<ScalarImageType>::New();
    reader->SetFileName(filename);
    reader->Update();
    ScalarImageType::Pointer img = reader->GetOutput();
    img->DisconnectPipeline();
    return img;
}

int testRepresenterForScalarImage(const std::string& datadir) {
    typedef itk::StandardImageRepresenter<float, 2> RepresenterType;
    typedef GenericRepresenterTest<RepresenterType> RepresenterTestType;

    const std::string referenceFilename = datadir + "/hand_images/hand-1.vtk";
    const std::string testDatasetFilename = datadir + "/hand_images/hand-2.vtk";

    RepresenterType::Pointer representer = RepresenterType::New();
    ScalarImageType::Pointer reference = loadScalarImage(referenceFilename);
    representer->SetReference(reference);

    // choose a test dataset, a point and its associate pixel value

    ScalarImageType::Pointer testDataset = loadScalarImage(testDatasetFilename);
    ScalarImageType::IndexType idx;
    idx.Fill(0);
    ScalarImageType::PointType testPt;
    reference->TransformIndexToPhysicalPoint(idx, testPt);
    ScalarImageType::PixelType testValue = testDataset->GetPixel(idx);

    RepresenterTestType representerTest(representer, testDataset, std::make_pair(testPt, testValue));

    return (representerTest.runAllTests() == true);
}


typedef itk::Image< itk::Vector<float,2>, 2 > VectorImageType;

VectorImageType::Pointer loadVectorImage(const std::string& filename) {
    itk::ImageFileReader<VectorImageType>::Pointer reader = itk::ImageFileReader<VectorImageType>::New();
    reader->SetFileName(filename);
    reader->Update();
    VectorImageType::Pointer img = reader->GetOutput();
    img->DisconnectPipeline();
    return img;
}

int testRepresenterForVectorImage(const std::string& datadir) {

    typedef itk::StandardImageRepresenter<itk::Vector<float, 2>, 2> RepresenterType;
    typedef GenericRepresenterTest<RepresenterType> RepresenterTestType;

    const std::string referenceFilename = datadir + "/hand_dfs/df-hand-1.vtk";
    const std::string testDatasetFilename = datadir + "/hand_dfs/df-hand-2.vtk";

    RepresenterType::Pointer representer = RepresenterType::New();
    VectorImageType::Pointer reference = loadVectorImage(referenceFilename);
    representer->SetReference(reference);

    // choose a test dataset, a point and its associate pixel value

    VectorImageType::Pointer testDataset = loadVectorImage(testDatasetFilename);
    VectorImageType::IndexType idx;
    idx.Fill(0);
    VectorImageType::PointType testPt;
    reference->TransformIndexToPhysicalPoint(idx, testPt);
    VectorImageType::PixelType testValue = testDataset->GetPixel(idx);

    RepresenterTestType representerTest(representer, testDataset, std::make_pair(testPt, testValue));

    return (representerTest.runAllTests() == true);
}


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " datadir" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::string datadir = std::string(argv[1]);
    bool resultScalarImage = testRepresenterForScalarImage(datadir);
    bool resultVectorImage = testRepresenterForVectorImage(datadir);

    if (resultScalarImage == true && resultVectorImage == true) {
        return EXIT_SUCCESS;
    } else {
        return EXIT_FAILURE;
    }
}


