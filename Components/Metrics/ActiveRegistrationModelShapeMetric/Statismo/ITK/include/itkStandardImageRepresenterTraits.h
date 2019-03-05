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

#ifndef __itkStandardImageRepresenterTraits_h
#define __itkStandardImageRepresenterTraits_h

#include "itkImage.h"
#include "itkVector.h"
#include "Representer.h"

namespace statismo {

template<>
struct RepresenterTraits<itk::Image<itk::Vector<double, 4u>, 4u> > {

    typedef itk::Image<itk::Vector<double, 4u>, 4u> VectorImageType;
    typedef VectorImageType::Pointer DatasetPointerType;
    typedef VectorImageType::Pointer DatasetConstPointerType;
    typedef VectorImageType::PointType PointType;
    typedef VectorImageType::PixelType ValueType;
};

template<>
struct RepresenterTraits<itk::Image<itk::Vector<double, 3u>, 3u> > {

    typedef itk::Image<itk::Vector<double, 3u>, 3u> VectorImageType;
    typedef VectorImageType::Pointer DatasetPointerType;
    typedef VectorImageType::Pointer DatasetConstPointerType;
    typedef VectorImageType::PointType PointType;
    typedef VectorImageType::PixelType ValueType;
};

template<>
struct RepresenterTraits<itk::Image<itk::Vector<double, 2u>, 2u> > {

    typedef itk::Image<itk::Vector<double, 2u>, 2u> VectorImageType;
    typedef VectorImageType::Pointer DatasetPointerType;
    typedef VectorImageType::Pointer DatasetConstPointerType;
    typedef VectorImageType::PointType PointType;
    typedef VectorImageType::PixelType ValueType;
};



template<>
struct RepresenterTraits<itk::Image<itk::Vector<float, 4u>, 4u> > {

    typedef itk::Image<itk::Vector<float, 4u>, 4u> VectorImageType;
    typedef VectorImageType::Pointer DatasetPointerType;
    typedef VectorImageType::Pointer DatasetConstPointerType;
    typedef VectorImageType::PointType PointType;
    typedef VectorImageType::PixelType ValueType;
};

template<>
struct RepresenterTraits<itk::Image<itk::Vector<float, 3u>, 3u> > {

    typedef itk::Image<itk::Vector<float, 3u>, 3u> VectorImageType;
    typedef VectorImageType::Pointer DatasetPointerType;
    typedef VectorImageType::Pointer DatasetConstPointerType;
    typedef VectorImageType::PointType PointType;
    typedef VectorImageType::PixelType ValueType;
};

template<>
struct RepresenterTraits<itk::Image<itk::Vector<float, 2u>, 2u> > {

    typedef itk::Image<itk::Vector<float, 2u>, 2u> VectorImageType;
    typedef VectorImageType::Pointer DatasetPointerType;
    typedef VectorImageType::Pointer DatasetConstPointerType;
    typedef VectorImageType::PointType PointType;
    typedef VectorImageType::PixelType ValueType;
};

template<>
struct RepresenterTraits<itk::Image<float, 4u> > {

    typedef itk::Image<float, 4u> ImageType;
    typedef ImageType::Pointer DatasetPointerType;
    typedef ImageType::Pointer DatasetConstPointerType;
    typedef ImageType::PointType PointType;
    typedef ImageType::PixelType ValueType;
};

template<>
struct RepresenterTraits<itk::Image<float, 3u> > {

    typedef itk::Image<float, 3u> ImageType;
    typedef ImageType::Pointer DatasetPointerType;
    typedef ImageType::Pointer DatasetConstPointerType;
    typedef ImageType::PointType PointType;
    typedef ImageType::PixelType ValueType;
};

template<>
struct RepresenterTraits<itk::Image<float, 2u> > {

    typedef itk::Image<float, 2u> ImageType;
    typedef ImageType::Pointer DatasetPointerType;
    typedef ImageType::Pointer DatasetConstPointerType;
    typedef ImageType::PointType PointType;
    typedef ImageType::PixelType ValueType;
};

template<>
struct RepresenterTraits<itk::Image<short, 4u> > {

    typedef itk::Image<short, 4u> ImageType;
    typedef ImageType::Pointer DatasetPointerType;
    typedef ImageType::Pointer DatasetConstPointerType;
    typedef ImageType::PointType PointType;
    typedef ImageType::PixelType ValueType;
};

template<>
struct RepresenterTraits<itk::Image<short, 3u> > {

    typedef itk::Image<short, 3u> ImageType;
    typedef ImageType::Pointer DatasetPointerType;
    typedef ImageType::Pointer DatasetConstPointerType;
    typedef ImageType::PointType PointType;
    typedef ImageType::PixelType ValueType;
};

template<>
struct RepresenterTraits<itk::Image<short, 2u> > {

    typedef itk::Image<float, 2u> ImageType;
    typedef ImageType::Pointer DatasetPointerType;
    typedef ImageType::Pointer DatasetConstPointerType;
    typedef ImageType::PointType PointType;
    typedef ImageType::PixelType ValueType;
};

template<>
struct RepresenterTraits<itk::Image<unsigned short, 4u> > {

    typedef itk::Image<unsigned short, 4u> ImageType;
    typedef ImageType::Pointer DatasetPointerType;
    typedef ImageType::Pointer DatasetConstPointerType;
    typedef ImageType::PointType PointType;
    typedef ImageType::PixelType ValueType;
};

template<>
struct RepresenterTraits<itk::Image<unsigned short, 3u> > {

    typedef itk::Image<unsigned short, 3u> ImageType;
    typedef ImageType::Pointer DatasetPointerType;
    typedef ImageType::Pointer DatasetConstPointerType;
    typedef ImageType::PointType PointType;
    typedef ImageType::PixelType ValueType;
};

template<>
struct RepresenterTraits<itk::Image<unsigned short, 2u> > {

    typedef itk::Image<float, 2u> ImageType;
    typedef ImageType::Pointer DatasetPointerType;
    typedef ImageType::Pointer DatasetConstPointerType;
    typedef ImageType::PointType PointType;
    typedef ImageType::PixelType ValueType;
};

template<>
struct RepresenterTraits<itk::Image<unsigned char, 4u> > {

    typedef itk::Image<char, 4u> ImageType;
    typedef ImageType::Pointer DatasetPointerType;
    typedef ImageType::Pointer DatasetConstPointerType;
    typedef ImageType::PointType PointType;
    typedef ImageType::PixelType ValueType;
};

template<>
struct RepresenterTraits<itk::Image<unsigned char, 3u> > {

    typedef itk::Image<char, 3u> ImageType;
    typedef ImageType::Pointer DatasetPointerType;
    typedef ImageType::Pointer DatasetConstPointerType;
    typedef ImageType::PointType PointType;
    typedef ImageType::PixelType ValueType;
};

template<>
struct RepresenterTraits<itk::Image<unsigned char, 2u> > {

    typedef itk::Image<short, 2u> ImageType;
    typedef ImageType::Pointer DatasetPointerType;
    typedef ImageType::Pointer DatasetConstPointerType;
    typedef ImageType::PointType PointType;
    typedef ImageType::PixelType ValueType;
};


} // namespace statismo

#endif
