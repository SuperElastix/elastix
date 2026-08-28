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

#ifndef __DOMAIN_H
#define __DOMAIN_H

#include <vector>

namespace statismo {

/**
 * This class represents the domain on which a statistical model is defined.
 * A domain is simply a list of points.
 */
//RB: enable adding / removing elements to avoid copying data?
template <typename PointType>
class Domain {
  public:
    typedef std::vector<PointType> DomainPointsListType;

    /**
     * Create an empty domain
     */
    Domain() {}

    /**
     * Create a new domain from the given list of points
     */
    Domain(const DomainPointsListType& domainPoints)
        : m_domainPoints(domainPoints) {}

    /** Returns a list of points that define the domain */
    const DomainPointsListType& GetDomainPoints() const {
        return m_domainPoints;
    }

    /** Returns the number of poitns of the domain */
    const unsigned GetNumberOfPoints() const {
        return m_domainPoints.size();
    }

  private:

    DomainPointsListType m_domainPoints;
};

} // namespace statismo

#endif
