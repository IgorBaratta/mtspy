#pragma once

namespace mtspy
{
    /// Simple implementation of and array view, similar to
    /// C++20's std::span. Remove when moving to c++ 20
    /// Note: not conforming with to the C++20 committee draft.
    template <typename T>
    class span
    {
        T *_data;
        std::size_t _size;

    public:
        /// Constructs a new span with data and size
        span(T *data, std::size_t size) noexcept : _data{data}, _size{size} {}

        /// Accesses an element of the sequence, do not check bounds
        T &operator[](int i) noexcept { return _data[i]; }

        /// Accesses an element of the sequence, do not check bounds
        T const &operator[](int i) const noexcept { return _data[i]; }

        /// Returns the number of elements in the span
        auto size() const noexcept { return _size; }

        /// Returns a pointer to the beginning of the span
        auto data() noexcept { return _data; }

        /// Returns an iterator to the beginning
        auto begin() noexcept { return _data; }

        /// Returns an iterator to the end
        auto end() noexcept { return _data + _size; }
    };

} // namespace mtspy
