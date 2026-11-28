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


#ifndef ITKSTATISMOIO_H_
#define ITKSTATISMOIO_H_

#include "itkStatisticalModel.h"
#include "StatismoIO.h"


namespace itk {

template <typename T>
class StatismoIO {
  private:
    //These tyedefs are only used internally and as such are marked as private
    typedef statismo::StatisticalModel<T>               StatisticalModelType;
    typedef itk::StatisticalModel<T>                    ITKStatisticalModelType;
    typedef typename ITKStatisticalModelType::Pointer   ITKStatisticalModelTypePointer;

  public:
    static ITKStatisticalModelTypePointer LoadStatisticalModel(
            typename StatisticalModelType::RepresenterType *representer, const std::string &filename,
            unsigned maxNumberOfPCAComponents = std::numeric_limits<unsigned>::max()) {
        try {
            ITKStatisticalModelTypePointer pModel = ITKStatisticalModelType::New();
            pModel->SetstatismoImplObj(statismo::IO<T>::LoadStatisticalModel(representer, filename, maxNumberOfPCAComponents));
            return pModel;
        } catch (const statismo::StatisticalModelException& e) {
            itkGenericExceptionMacro(<< e.what());
        }
    }

    static ITKStatisticalModelTypePointer LoadStatisticalModel(
            typename ITKStatisticalModelType::RepresenterType *representer, const H5::Group &modelRoot,
            unsigned maxNumberOfPCAComponents = std::numeric_limits<unsigned>::max()) {
        try {
            ITKStatisticalModelTypePointer pModel = ITKStatisticalModelType::New();
            pModel->SetstatismoImplObj(statismo::IO<T>::LoadStatisticalModel(representer, modelRoot, maxNumberOfPCAComponents));
            return pModel;
        } catch (const statismo::StatisticalModelException& e) {
            itkGenericExceptionMacro(<< e.what());
        }
    }

    static void SaveStatisticalModel(const ITKStatisticalModelType *const model, const std::string &filename) {
        try {
            statismo::IO<T>::SaveStatisticalModel(model->GetstatismoImplObj(), filename);
        } catch (const statismo::StatisticalModelException& e) {
            itkGenericExceptionMacro(<< e.what());
        }
    }

    static void SaveStatisticalModel(const ITKStatisticalModelType *model, const H5::Group &modelRoot) {
        try {
            statismo::IO<T>::SaveStatisticalModel(model->GetstatismoImplObj(), modelRoot);
        } catch (const statismo::StatisticalModelException& e) {
            itkGenericExceptionMacro(<< e.what());
        }
    }
};

} // namespace itk

#endif /* ITKSTATISMOIO_H_ */
